import os
import io
import re
import json
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import pytesseract

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from db import SessionLocal, engine, Base
from models import Student, HomeworkAssignment, HomeworkImage, Submission, Result, AuditLog



# =========================
# TESSERACT CONFIG (WINDOWS)
# =========================
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"


# =========================
# ANSWER KEY CONFIG
# =========================
ANSWER_KEY_PATH = "answer_key.json"


def load_answer_key(homework_id: str):
    """
    answer_key.json format:
    {
      "hw01": {"questions":[...]},
      "hw02": {"questions":[...]}
    }
    """
    try:
        with open(ANSWER_KEY_PATH, "r", encoding="utf-8") as f:
            all_keys = json.load(f)

        # DEBUG (keep for now)
        print("Available homework_ids in key:", list(all_keys.keys()))
        print("Requested homework_id:", homework_id)

        return all_keys.get(homework_id)  # None if not found

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="answer_key.json file missing")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="answer_key.json is invalid JSON")


# =========================
# OCR
# =========================
def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image / OCR failed: {e}")


# =========================
# SEGMENTATION (Q1/Q2...)
# =========================
def segment_answers_by_question(text: str) -> Dict[str, str]:
    cleaned = text.replace("\r", "\n")
    pattern = re.compile(r"\bQ\s*([0-9]+)\s*[\.\:\-]?", re.IGNORECASE)
    matches = list(pattern.finditer(cleaned))

    segments: Dict[str, str] = {}
    for i, m in enumerate(matches):
        qnum = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        answer_block = cleaned[start:end].strip()
        segments[f"Q{qnum}"] = answer_block

    return segments


# =========================
# CLEANING (remove UI junk)
# =========================
def clean_student_answer(raw_block: str) -> str:
    lines = [ln.strip() for ln in raw_block.splitlines() if ln.strip()]
    if not lines:
        return ""

    # remove question line if it looks like a question
    if lines[0].endswith("?") or lines[0].lower().startswith(("what ", "why ", "how ", "define ")):
        lines = lines[1:]

    cleaned_lines = []
    for ln in lines:
        low = ln.lower()

        # skip common editor/UI noise
        if low.startswith("ln") and "col" in low:
            continue
        if "plain text" in low:
            continue
        if "c:\\users" in low or "cusers" in low:
            continue
        if low.endswith("%"):  # like 100%
            continue

        # skip lines mostly symbols/numbers
        letters = sum(ch.isalpha() for ch in ln)
        if letters < 3 and len(ln) > 3:
            continue

        cleaned_lines.append(ln)

    return " ".join(cleaned_lines).strip()


# =========================
# SIMILARITY (TF-IDF cosine)
# =========================
def semantic_similarity(a: str, b: str) -> float:
    a = a.strip().lower()
    b = b.strip().lower()
    if not a or not b:
        return 0.0
    vect = TfidfVectorizer().fit([a, b])
    X = vect.transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0][0])


def best_sentence_similarity(student_text: str, expected_text: str) -> float:
    parts = re.split(r"[.\n]+", student_text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return 0.0
    scores = [semantic_similarity(p, expected_text) for p in parts]
    return max(scores)


# =========================
# VALIDATION
# =========================
def validate_against_key(homework_id: str, segmented_answers: Dict[str, str]) -> Dict[str, Any]:
    key = load_answer_key(homework_id)

    # ✅ Allow ANY homework_id: if key missing, do OCR+segmentation only
    if key is None:
        return {
            "status": "NO_ANSWER_KEY",
            "total": 0,
            "correct": 0,
            "overall_score": None,
            "per_question": [],
            "message": f"No answer key found for homework_id={homework_id}. Stored OCR + segmentation only."
        }

    results = []
    correct = 0

    for q in key.get("questions", []):
        qid = q.get("qid", "").strip()
        qtype = q.get("type", "text")
        expected_raw = q.get("answer", "")

        raw_student = segmented_answers.get(qid, "").strip()
        student_clean = clean_student_answer(raw_student)

        # Missing
        if not student_clean:
            results.append({
                "qid": qid,
                "expected": expected_raw,
                "student_answer": raw_student,
                "cleaned_answer_used_for_check": student_clean,
                "is_correct": False,
                "confidence": 0.0,
                "reason": "missing"
            })
            continue

        # Numeric tolerance
        if qtype == "numeric":
            try:
                student_num = float(student_clean)
                expected_num = float(expected_raw)
                tol = float(q.get("tolerance", 0.0))

                is_correct = abs(student_num - expected_num) <= tol
                confidence = 1.0 if is_correct else 0.0

                if is_correct:
                    correct += 1

                results.append({
                    "qid": qid,
                    "expected": expected_raw,
                    "student_answer": raw_student,
                    "cleaned_answer_used_for_check": student_clean,
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "reason": "tolerance_check",
                    "tolerance": tol
                })
                continue
            except:
                results.append({
                    "qid": qid,
                    "expected": expected_raw,
                    "student_answer": raw_student,
                    "cleaned_answer_used_for_check": student_clean,
                    "is_correct": False,
                    "confidence": 0.0,
                    "reason": "numeric_parse_failed"
                })
                continue

        # Text: exact OR best-sentence semantic match
        expected_text = str(expected_raw).strip()
        student_text = student_clean.strip()

        if student_text.lower() == expected_text.lower():
            is_correct = True
            confidence = 1.0
            reason = "exact_match"
        else:
            sim = best_sentence_similarity(student_text, expected_text)
            confidence = sim
            is_correct = sim >= 0.80
            reason = "semantic_match" if is_correct else "semantic_mismatch"

        if is_correct:
            correct += 1

        results.append({
            "qid": qid,
            "expected": expected_raw,
            "student_answer": raw_student,
            "cleaned_answer_used_for_check": student_clean,
            "is_correct": is_correct,
            "confidence": confidence,
            "reason": reason
        })

    total = len(key.get("questions", []))
    return {
        "status": "GRADED",
        "total": total,
        "correct": correct,
        "overall_score": (correct / total) if total else 0.0,
        "per_question": results
    }


# =========================
# FASTAPI APP + DB TABLES
# =========================
app = FastAPI(title="Homework Validation System")
Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/submit")
async def submit_homework(
    student_id: str = Form(...),
    homework_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if not student_id.strip() or not homework_id.strip():
        raise HTTPException(status_code=400, detail="student_id and homework_id are required")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")

    db = SessionLocal()

    try:
        # -----------------------------
        # 1) UPSERT Student
        # -----------------------------
        student = db.query(Student).filter(Student.student_id == student_id).first()
        if not student:
            student = Student(student_id=student_id)
            db.add(student)
            db.commit()
            db.refresh(student)

        # -----------------------------
        # 2) UPSERT HomeworkAssignment
        # -----------------------------
        hw = db.query(HomeworkAssignment).filter(HomeworkAssignment.homework_id == homework_id).first()
        if not hw:
            hw = HomeworkAssignment(homework_id=homework_id)
            db.add(hw)
            db.commit()
            db.refresh(hw)

        # -----------------------------
        # 3) Create Submission
        # -----------------------------
        submission = Submission(
            student_id=student_id,
            homework_id=homework_id,
            student_ref_id=student.id,
            homework_ref_id=hw.id,
            status="processed"
        )
        db.add(submission)
        db.commit()
        db.refresh(submission)

        extracted_data = []

        # -----------------------------
        # 4) For each image:
        #    save image row + OCR + segment + validate + result row
        # -----------------------------
        for img in images:
            # store image metadata row (required by plan)
            img_row = HomeworkImage(
                submission_id=submission.id,
                filename=img.filename,
                content_type=img.content_type
            )
            db.add(img_row)
            db.commit()

            content = await img.read()

            # OCR
            text = extract_text_from_image(content)

            # Segment
            segmented = segment_answers_by_question(text)

            # Reject invalid submissions (no Q1/Q2...)
            if not segmented:
                raise HTTPException(
                    status_code=400,
                    detail=f"No question numbers detected in {img.filename}. Expected Q1/Q2 format."
                )

            # Validate
            validation_report = validate_against_key(homework_id, segmented)

            # Save result row
            result_row = Result(
                submission_id=submission.id,
                filename=img.filename,
                extracted_text=text,
                segmented_answers_json=json.dumps(segmented, ensure_ascii=False),
                validation_json=json.dumps(validation_report, ensure_ascii=False)
            )
            db.add(result_row)
            db.commit()

            extracted_data.append({
                "filename": img.filename,
                "extracted_text": text,
                "segmented_answers": segmented,
                "validation": validation_report
            })

        db.add(AuditLog(submission_id=submission.id, level="INFO", message="Submission processed successfully"))
        db.commit()

        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "submission_id": submission.id,
            "extracted_data": extracted_data,
            "message": "OCR completed. Next step: answer extraction + validation."
        }

    except HTTPException as he:
        # mark submission failed if created
        try:
            if "submission" in locals():
                submission.status = "failed"
                db.add(submission)
                db.add(AuditLog(submission_id=submission.id, level="ERROR", message=str(he.detail)))
                db.commit()
        except:
            pass

        raise

    except Exception as e:
        try:
            if "submission" in locals():
                submission.status = "failed"
                db.add(submission)
                db.add(AuditLog(submission_id=submission.id, level="ERROR", message=str(e)))
                db.commit()
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    finally:
        db.close()



@app.get("/submissions")
def list_submissions():
    db = SessionLocal()
    items = db.query(Submission).order_by(Submission.id.desc()).limit(20).all()
    db.close()
    return [
        {
            "id": s.id,
            "student_id": s.student_id,
            "homework_id": s.homework_id,
            "status": s.status,
            "created_at": str(s.created_at)
        }
        for s in items
    ]


@app.post("/admin/answer-key")
def upsert_answer_key(homework_id: str = Form(...), answer_key_json: str = Form(...)):
    """
    Upload or update answer key for a given homework_id.
    answer_key_json should be JSON string like:
    {"questions":[{"qid":"Q1","type":"text","answer":"..."}]}
    """
    try:
        new_key = json.loads(answer_key_json)
        if "questions" not in new_key or not isinstance(new_key["questions"], list):
            raise HTTPException(status_code=400, detail="Invalid key format: must contain 'questions' list")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="answer_key_json must be valid JSON")

    # Load existing file (or create new)
    try:
        with open(ANSWER_KEY_PATH, "r", encoding="utf-8") as f:
            all_keys = json.load(f)
    except FileNotFoundError:
        all_keys = {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="answer_key.json is invalid JSON")

    all_keys[homework_id] = new_key

    with open(ANSWER_KEY_PATH, "w", encoding="utf-8") as f:
        json.dump(all_keys, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "message": f"Answer key saved for {homework_id}", "questions": len(new_key["questions"])}


@app.post("/admin/regrade/{submission_id}")
def regrade_submission(submission_id: int):
    db = SessionLocal()

    sub = db.query(Submission).filter(Submission.id == submission_id).first()
    if not sub:
        db.close()
        raise HTTPException(status_code=404, detail="Submission not found")

    results = db.query(Result).filter(Result.submission_id == submission_id).all()
    if not results:
        db.close()
        raise HTTPException(status_code=404, detail="No results found for this submission")

    updated = 0
    for r in results:
        try:
            segmented = json.loads(r.segmented_answers_json or "{}")
            validation_report = validate_against_key(sub.homework_id, segmented)

            r.validation_json = json.dumps(validation_report, ensure_ascii=False)
            db.add(r)
            updated += 1
        except Exception as e:
            db.add(AuditLog(submission_id=submission_id, level="ERROR", message=f"Regrade failed: {e}"))

    db.add(AuditLog(submission_id=submission_id, level="INFO", message=f"Regraded {updated} result rows"))
    db.commit()
    db.close()

    return {"status": "ok", "submission_id": submission_id, "updated_results": updated}
