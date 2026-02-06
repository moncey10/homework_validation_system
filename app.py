import os
import re
import io
import json
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import pytesseract

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sqlalchemy import and_

from db import SessionLocal, engine, Base
from models import Student, HomeworkAssignment, Submission, HomeworkImage, Result, AuditLog


# =========================================================
# TESSERACT SETUP (Windows path included)
# =========================================================
# If you are on Windows and Tesseract is installed here, this works immediately:
DEFAULT_TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# You can override by setting env var:
# PowerShell:
#   setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
# CMD:
#   setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

if os.name == "nt":
    # On Windows, try env override first; else fallback to default.
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD or DEFAULT_TESSERACT_WIN
else:
    # On Linux/Mac, pytesseract usually finds tesseract if installed.
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# =========================================================
# APP INIT
# =========================================================
app = FastAPI(title="Homework Validation System")
Base.metadata.create_all(bind=engine)


# =========================================================
# OCR
# =========================================================
def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        return (pytesseract.image_to_string(img, lang="eng") or "").strip()
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Tesseract OCR not found.\n"
                "Fix: Install Tesseract and ensure path is correct.\n"
                f"Current tesseract_cmd: {pytesseract.pytesseract.tesseract_cmd}\n"
                "Windows default expected: C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n"
                "Or set env var TESSERACT_CMD to your tesseract.exe path."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


# =========================================================
# FLEX SEGMENTATION (Q/STEP/ITEM/FULL_TEXT)
# =========================================================
def _segment_by_regex(text: str, pattern: re.Pattern, label_builder) -> Dict[str, str]:
    cleaned = (text or "").replace("\r", "\n")
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return {}

    segs: Dict[str, str] = {}
    for i, m in enumerate(matches):
        label = label_builder(m)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        segs[label] = cleaned[start:end].strip()
    return segs


def segment_text_flexibly(text: str) -> Dict[str, str]:
    """
    Detects:
      - Q1 / Q2 / Q3
      - Step 1 / Step 3A / Step 3B
      - 1) / 2) / 3)   OR   1. / 2. / 3.
    If nothing found -> FULL_TEXT fallback (no errors).
    """

    # Q style
    q_pat = re.compile(r"\bQ\s*([0-9]+)\s*[\.\:\-]?\s*", re.IGNORECASE)
    segs = _segment_by_regex(text, q_pat, lambda m: f"Q{m.group(1)}")
    if segs:
        return segs

    # Step style (Step 3A etc.)
    step_pat = re.compile(r"\bStep\s*([0-9]+[A-Za-z]?)\s*[\.\:\-]?\s*", re.IGNORECASE)
    segs = _segment_by_regex(text, step_pat, lambda m: f"STEP{m.group(1).upper()}")
    if segs:
        return segs

    # Numbered list style (must be line-start)
    num_pat = re.compile(r"(?m)^\s*([0-9]{1,2})\s*[\)\.]\s+")
    segs = _segment_by_regex(text, num_pat, lambda m: f"ITEM{m.group(1)}")
    if segs:
        return segs

    # Fallback: accept diagram/paragraph pages too
    full = (text or "").strip()
    return {"FULL_TEXT": full} if full else {}


def segmentation_type(segments: Dict[str, str]) -> str:
    keys = list(segments.keys())
    if any(k.startswith("Q") for k in keys):
        return "Q"
    if any(k.startswith("STEP") for k in keys):
        return "STEP"
    if any(k.startswith("ITEM") for k in keys):
        return "ITEM"
    return "FULL_TEXT"


# =========================================================
# SIMILARITY + GRADING USING TEACHER REFERENCE
# =========================================================
def cosine_sim(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    vect = TfidfVectorizer().fit([a, b])
    X = vect.transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0][0])


def grade_using_teacher_reference(
    teacher_segments: Dict[str, str],
    student_segments: Dict[str, str],
    threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Grades per segment label (Q1/STEP1/ITEM1/FULL_TEXT) using cosine similarity.
    No answer_key.json required.
    """

    # If one side is FULL_TEXT and the other has STEP/Q etc, grading is not meaningful.
    t_type = segmentation_type(teacher_segments)
    s_type = segmentation_type(student_segments)

    # If either side is empty
    if not teacher_segments:
        return {
            "mode": "TEACHER_REFERENCE_GRADING",
            "status": "NO_TEACHER_TEXT",
            "message": "Teacher OCR text is empty; cannot grade.",
            "total": 0,
            "correct": 0,
            "overall_score": None,
            "per_item": [],
        }

    if not student_segments:
        return {
            "mode": "TEACHER_REFERENCE_GRADING",
            "status": "NO_STUDENT_TEXT",
            "message": "Student OCR text is empty; cannot grade.",
            "total": 0,
            "correct": 0,
            "overall_score": None,
            "per_item": [],
        }

    # If teacher is FULL_TEXT but student is STEP/Q/ITEM, we can still do whole-text compare.
    # If teacher is STEP/Q/ITEM but student is FULL_TEXT, we can also do whole-text compare.
    # This avoids failing when layouts differ.
    if (t_type == "FULL_TEXT" and s_type != "FULL_TEXT") or (s_type == "FULL_TEXT" and t_type != "FULL_TEXT"):
        t_full = teacher_segments.get("FULL_TEXT", "")
        s_full = student_segments.get("FULL_TEXT", "")
        # If missing FULL_TEXT on one side, join all segments
        if not t_full:
            t_full = "\n".join([v for _, v in sorted(teacher_segments.items()) if v])
        if not s_full:
            s_full = "\n".join([v for _, v in sorted(student_segments.items()) if v])

        sim = cosine_sim(s_full, t_full)
        return {
            "mode": "TEACHER_REFERENCE_GRADING",
            "status": "WHOLE_TEXT_COMPARE",
            "threshold": threshold,
            "teacher_segmentation_type": t_type,
            "student_segmentation_type": s_type,
            "total": 1,
            "correct": 1 if sim >= threshold else 0,
            "overall_score": 1.0 if sim >= threshold else 0.0,
            "per_item": [{
                "label": "FULL_TEXT",
                "similarity": sim,
                "is_correct": sim >= threshold,
                "reason": "match" if sim >= threshold else "mismatch",
            }],
        }

    # Normal case: compare per label
    labels = sorted(set(teacher_segments.keys()) | set(student_segments.keys()))
    per_item = []
    correct = 0
    total = 0

    for label in labels:
        t = (teacher_segments.get(label) or "").strip()
        s = (student_segments.get(label) or "").strip()

        if not t and not s:
            continue

        total += 1

        if not t:
            per_item.append({
                "label": label,
                "status": "NO_TEACHER_SEGMENT",
                "similarity": None,
                "is_correct": None,
            })
            continue

        if not s:
            per_item.append({
                "label": label,
                "status": "MISSING_STUDENT_ANSWER",
                "similarity": 0.0,
                "is_correct": False,
            })
            continue

        sim = cosine_sim(s, t)
        is_ok = sim >= threshold
        if is_ok:
            correct += 1

        per_item.append({
            "label": label,
            "similarity": sim,
            "is_correct": is_ok,
            "reason": "match" if is_ok else "mismatch",
        })

    return {
        "mode": "TEACHER_REFERENCE_GRADING",
        "status": "PER_SEGMENT_COMPARE",
        "threshold": threshold,
        "teacher_segmentation_type": t_type,
        "student_segmentation_type": s_type,
        "total": total,
        "correct": correct,
        "overall_score": (correct / total) if total else None,
        "per_item": per_item,
    }


def make_unique_filename(original: str, used: set, idx: int) -> str:
    original = (original or "").strip() or f"image_{idx}.png"
    name = original
    if name in used:
        name = f"{idx}_{original}"
    used.add(name)
    return name


# =========================================================
# ROUTES
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/tesseract")
def debug_tesseract():
    try:
        version = str(pytesseract.get_tesseract_version())
        return {
            "tesseract": "OK",
            "version": version,
            "tesseract_cmd": pytesseract.pytesseract.tesseract_cmd,
        }
    except Exception as e:
        return {
            "tesseract": "NOT_FOUND",
            "error": str(e),
            "tesseract_cmd": pytesseract.pytesseract.tesseract_cmd,
        }


@app.post("/teacher/homework")
async def upload_teacher_homework(
    homework_id: str = Form(...),
    teacher_image: UploadFile = File(...),
):
    """
    Upload teacher reference "answer" page/image for a homework_id.
    Student submissions will be graded against this (no answer_key.json needed).
    """
    db = SessionLocal()
    try:
        homework_id = homework_id.strip()
        if not homework_id:
            raise HTTPException(status_code=400, detail="homework_id is required")

        # ensure Homework row
        hw = db.query(HomeworkAssignment).filter(HomeworkAssignment.homework_id == homework_id).first()
        if not hw:
            hw = HomeworkAssignment(homework_id=homework_id)
            db.add(hw)
            db.commit()
            db.refresh(hw)

        content = await teacher_image.read()
        teacher_text = extract_text_from_image(content)

        existing = db.query(HomeworkImage).filter(
            and_(HomeworkImage.homework_id == homework_id, HomeworkImage.role == "teacher")
        ).first()

        if existing:
            existing.filename = teacher_image.filename
            existing.content_type = teacher_image.content_type
            existing.ocr_text = teacher_text
            db.add(existing)
        else:
            db.add(HomeworkImage(
                submission_id=None,
                homework_id=homework_id,
                role="teacher",
                filename=teacher_image.filename,
                content_type=teacher_image.content_type,
                ocr_text=teacher_text,
            ))

        db.add(AuditLog(submission_id=None, level="INFO", message=f"Teacher reference uploaded for {homework_id}"))
        db.commit()

        return {
            "homework_id": homework_id,
            "teacher_filename": teacher_image.filename,
            "teacher_segmentation_type": segmentation_type(segment_text_flexibly(teacher_text)),
            "message": "Teacher reference stored (OCR done).",
        }

    except HTTPException:
        raise
    except Exception as e:
        db.add(AuditLog(submission_id=None, level="ERROR", message=f"Teacher upload failed: {e}"))
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/submit")
async def submit_homework(
    student_id: str = Form(...),
    homework_id: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """
    Student submission:
    - OCR each image
    - Flexible segmentation (Q/STEP/ITEM/FULL_TEXT)
    - Grade against teacher reference for the same homework_id
    """
    db = SessionLocal()
    submission = None

    try:
        student_id = student_id.strip()
        homework_id = homework_id.strip()

        if not student_id or not homework_id:
            raise HTTPException(status_code=400, detail="student_id and homework_id are required")
        if not images:
            raise HTTPException(status_code=400, detail="At least one image is required")

        # Load teacher reference once
        teacher_row = db.query(HomeworkImage).filter(
            and_(HomeworkImage.homework_id == homework_id, HomeworkImage.role == "teacher")
        ).first()
        teacher_text = (teacher_row.ocr_text or "").strip() if teacher_row else ""
        has_teacher_reference = bool(teacher_text)
        teacher_segments = segment_text_flexibly(teacher_text) if has_teacher_reference else {}

        # ensure Student
        student = db.query(Student).filter(Student.student_id == student_id).first()
        if not student:
            student = Student(student_id=student_id)
            db.add(student)
            db.commit()
            db.refresh(student)

        # ensure Homework
        hw = db.query(HomeworkAssignment).filter(HomeworkAssignment.homework_id == homework_id).first()
        if not hw:
            hw = HomeworkAssignment(homework_id=homework_id)
            db.add(hw)
            db.commit()
            db.refresh(hw)

        # create Submission
        submission = Submission(
            student_id=student_id,
            homework_id=homework_id,
            student_ref_id=student.id,
            homework_ref_id=hw.id,
            status="processed",
        )
        db.add(submission)
        db.commit()
        db.refresh(submission)

        used_names = set()
        extracted_data = []

        for idx, img in enumerate(images, start=1):
            raw_filename = img.filename
            safe_filename = make_unique_filename(raw_filename, used_names, idx)

            content = await img.read()
            student_text = extract_text_from_image(content)
            student_segments = segment_text_flexibly(student_text)

            # grade using teacher reference (no answer_key needed)
            if has_teacher_reference:
                grading = grade_using_teacher_reference(
                    teacher_segments=teacher_segments,
                    student_segments=student_segments,
                    threshold=0.75
                )
            else:
                grading = {
                    "mode": "TEACHER_REFERENCE_GRADING",
                    "status": "NO_TEACHER_REFERENCE",
                    "message": "No teacher reference uploaded for this homework_id. Upload using /teacher/homework.",
                    "total": 0,
                    "correct": 0,
                    "overall_score": None,
                    "per_item": [],
                }

            # Save OCR + results
            db.add(HomeworkImage(
                submission_id=submission.id,
                homework_id=homework_id,
                role="student",
                filename=safe_filename,
                content_type=img.content_type,
                ocr_text=student_text,
            ))
            db.commit()

            validation_payload = {
                "grading": grading,
                "teacher_reference_found": has_teacher_reference,
                "teacher_filename": teacher_row.filename if teacher_row else None,
                "segmentation_type": segmentation_type(student_segments),
            }

            db.add(Result(
                submission_id=submission.id,
                filename=safe_filename,
                extracted_text=student_text,
                segmented_answers_json=json.dumps(student_segments, ensure_ascii=False),
                validation_json=json.dumps(validation_payload, ensure_ascii=False),
            ))
            db.commit()

            extracted_data.append({
                "original_filename": raw_filename,
                "stored_filename": safe_filename,
                "segmented_answers": student_segments,
                "validation": validation_payload,
            })

        db.add(AuditLog(submission_id=submission.id, level="INFO", message="Submission processed"))
        db.commit()

        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "submission_id": submission.id,
            "files_processed": len(images),
            "teacher_reference_found": has_teacher_reference,
            "extracted_data": extracted_data,
            "message": "OCR done. Graded using teacher reference (no answer key needed).",
        }

    except HTTPException as he:
        if submission is not None:
            submission.status = "failed"
            db.add(submission)
            db.add(AuditLog(submission_id=submission.id, level="ERROR", message=str(he.detail)))
            db.commit()
        raise

    except Exception as e:
        if submission is not None:
            submission.status = "failed"
            db.add(submission)
            db.add(AuditLog(submission_id=submission.id, level="ERROR", message=str(e)))
            db.commit()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    finally:
        db.close()
