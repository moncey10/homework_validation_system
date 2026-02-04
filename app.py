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
from models import Submission, Result, AuditLog


# =========================
# TESSERACT CONFIG (WINDOWS)
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"


# =========================
# ANSWER KEY CONFIG
# =========================
ANSWER_KEY_PATH = "answer_key.json"


def load_answer_key() -> Dict[str, Any]:
    try:
        with open(ANSWER_KEY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Missing {ANSWER_KEY_PATH} file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load answer key: {e}")


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
    key = load_answer_key()

    if key.get("homework_id") != homework_id:
        raise HTTPException(status_code=400, detail="Answer key not found for this homework_id")

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
    submission = Submission(student_id=student_id, homework_id=homework_id, status="processed")
    db.add(submission)
    db.commit()
    db.refresh(submission)

    extracted_data = []

    try:
        for img in images:
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

    except HTTPException as he:
        submission.status = "failed"
        db.add(submission)
        db.add(AuditLog(submission_id=submission.id, level="ERROR", message=str(he.detail)))
        db.commit()
        db.close()
        raise

    except Exception as e:
        submission.status = "failed"
        db.add(submission)
        db.add(AuditLog(submission_id=submission.id, level="ERROR", message=str(e)))
        db.commit()
        db.close()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    db.close()

    return {
        "student_id": student_id,
        "homework_id": homework_id,
        "extracted_data": extracted_data,
        "message": "OCR completed. Next step: answer extraction + validation."
    }


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
