import os
import io
import re
from typing import Dict, Any, List, Tuple, Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import pytesseract

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# ✅ TESSERACT PATH
# =========================================================
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


# =========================================================
# ERP CONFIG
# =========================================================
ERP_BASE = "https://erp.triz.co.in/lms_data"
STORAGE_BASE = "https://erp.triz.co.in/storage/student/"


# =========================================================
# APP INIT
# =========================================================
app = FastAPI(title="Homework Validation System (ERP Teacher Auto)")


# =========================================================
# OCR
# =========================================================
def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        return (pytesseract.image_to_string(img, lang="eng") or "").strip()
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Tesseract OCR not found.\n"
                f"tesseract_cmd: {pytesseract.pytesseract.tesseract_cmd}\n"
                "Fix: Install Tesseract or set correct path in app.py."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


# =========================================================
# SEGMENTATION
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
    q_pat = re.compile(r"\bQ\s*([0-9]+)\s*[\.\:\-]?\s*", re.IGNORECASE)
    segs = _segment_by_regex(text, q_pat, lambda m: f"Q{m.group(1)}")
    if segs:
        return segs

    step_pat = re.compile(r"\bStep\s*([0-9]+[A-Za-z]?)\s*[\.\:\-]?\s*", re.IGNORECASE)
    segs = _segment_by_regex(text, step_pat, lambda m: f"STEP{m.group(1).upper()}")
    if segs:
        return segs

    num_pat = re.compile(r"(?m)^\s*([0-9]{1,2})\s*[\)\.]\s+")
    segs = _segment_by_regex(text, num_pat, lambda m: f"ITEM{m.group(1)}")
    if segs:
        return segs

    full = (text or "").strip()
    return {"FULL_TEXT": full} if full else {}


def segmentation_type(segs: Dict[str, str]) -> str:
    keys = list(segs.keys())
    if any(k.startswith("Q") for k in keys):
        return "Q"
    if any(k.startswith("STEP") for k in keys):
        return "STEP"
    if any(k.startswith("ITEM") for k in keys):
        return "ITEM"
    return "FULL_TEXT"


# =========================================================
# SIMILARITY + GRADING
# =========================================================
def cosine_sim(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    vec = TfidfVectorizer().fit([a, b])
    X = vec.transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0][0])


def grade_using_teacher_reference(
    teacher_segments: Dict[str, str],
    student_segments: Dict[str, str],
    threshold: float = 0.75
) -> Dict[str, Any]:
    t_type = segmentation_type(teacher_segments)
    s_type = segmentation_type(student_segments)

    if not teacher_segments:
        return {"mode": "TEACHER_REFERENCE_GRADING", "status": "NO_TEACHER_TEXT", "overall_score": None, "per_item": []}
    if not student_segments:
        return {"mode": "TEACHER_REFERENCE_GRADING", "status": "NO_STUDENT_TEXT", "overall_score": None, "per_item": []}

    # If one FULL_TEXT and other segmented -> whole compare
    if (t_type == "FULL_TEXT" and s_type != "FULL_TEXT") or (s_type == "FULL_TEXT" and t_type != "FULL_TEXT"):
        t_full = teacher_segments.get("FULL_TEXT") or "\n".join(teacher_segments.values())
        s_full = student_segments.get("FULL_TEXT") or "\n".join(student_segments.values())
        sim = cosine_sim(s_full, t_full)
        ok = sim >= threshold
        return {
            "mode": "TEACHER_REFERENCE_GRADING",
            "status": "WHOLE_TEXT_COMPARE",
            "threshold": threshold,
            "teacher_segmentation_type": t_type,
            "student_segmentation_type": s_type,
            "overall_score": 1.0 if ok else 0.0,
            "per_item": [{"label": "FULL_TEXT", "similarity": sim, "is_correct": ok, "reason": "match" if ok else "mismatch"}],
        }

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
            per_item.append({"label": label, "status": "NO_TEACHER_SEGMENT", "similarity": None, "is_correct": None})
            continue
        if not s:
            per_item.append({"label": label, "status": "MISSING_STUDENT_ANSWER", "similarity": 0.0, "is_correct": False})
            continue

        sim = cosine_sim(s, t)
        ok = sim >= threshold
        if ok:
            correct += 1
        per_item.append({"label": label, "similarity": sim, "is_correct": ok, "reason": "match" if ok else "mismatch"})

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


# =========================================================
# ERP HELPERS (ROBUST TEACHER IMAGE PICK)
# =========================================================
def _erp_get(params: dict) -> list:
    r = requests.get(ERP_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="ERP returned invalid JSON (not a list).")
    return data


def fetch_student_record(homework_id: int, student_id: int) -> Dict[str, Any]:
    data = _erp_get({"table": "homework", "filters[id]": homework_id, "filters[student_id]": student_id})
    if len(data) == 0:
        raise HTTPException(status_code=404, detail="No ERP record found for this homework_id + student_id")
    return data[0]


def fetch_teacher_image_filename(homework_id: int, student_id: int) -> str:
    """
    Try best-effort:
      1) homework_id + student_id
      2) homework_id only
    Return first non-empty 'image'.
    """
    # 1) try with student filter
    try:
        data1 = _erp_get({"table": "homework", "filters[id]": homework_id, "filters[student_id]": student_id})
        if len(data1) > 0:
            img = (data1[0].get("image") or "").strip()
            if img:
                return img
    except Exception:
        pass

    # 2) try without student filter
    data2 = _erp_get({"table": "homework", "filters[id]": homework_id})
    if len(data2) == 0:
        raise HTTPException(status_code=404, detail="No ERP record found for this homework_id (teacher)")
    img2 = (data2[0].get("image") or "").strip()
    if img2:
        return img2

    raise HTTPException(
        status_code=400,
        detail="Teacher image missing in ERP for this homework_id (field 'image' is empty)."
    )


def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


# Teacher cache: homework_id -> (teacher_filename, teacher_text)
_TEACHER_CACHE: Dict[int, Tuple[str, str]] = {}


def get_teacher_text_cached(homework_id: int, teacher_filename: str, teacher_url: str) -> Tuple[str, bool]:
    cached = _TEACHER_CACHE.get(homework_id)
    if cached and cached[0] == teacher_filename and cached[1].strip():
        return cached[1], True
    teacher_bytes = download_bytes(teacher_url)
    teacher_text = extract_text_from_image(teacher_bytes)
    _TEACHER_CACHE[homework_id] = (teacher_filename, teacher_text)
    return teacher_text, False


# =========================================================
# ROUTES
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/submit")
async def submit(
    student_id: int = Form(...),
    homework_id: int = Form(...),          # ERP homework record id like 2249927
    images: List[UploadFile] = File(...),  # student uploads
    threshold: float = Form(0.75),
):
    if not images:
        raise HTTPException(status_code=400, detail="At least one student image is required")

    # Student metadata (completion_status + submission_remarks)
    try:
        student_rec = fetch_student_record(homework_id, student_id)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"ERP student fetch failed: {e}")

    completion_status = student_rec.get("completion_status")
    # submission_remarks = student_rec.get("submission_remarks")

    # Teacher image filename (robust)
    try:
        teacher_filename = fetch_teacher_image_filename(homework_id, student_id)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"ERP teacher fetch failed: {e}")

    teacher_url = STORAGE_BASE + teacher_filename

    # Teacher OCR (cached)
    teacher_text, teacher_from_cache = get_teacher_text_cached(homework_id, teacher_filename, teacher_url)
    teacher_segments = segment_text_flexibly(teacher_text)

    extracted_data = []
    for img in images:
        student_bytes = await img.read()
        student_text = extract_text_from_image(student_bytes)
        student_segments = segment_text_flexibly(student_text)

        grading = grade_using_teacher_reference(
            teacher_segments=teacher_segments,
            student_segments=student_segments,
            threshold=float(threshold),
        )
        overall_score = grading.get("overall_score") or 0

        if overall_score >= threshold:
            submission_remarks = "Answer is complete and correct"
        else:
            submission_remarks = "Attempt appreciated — revise topic"


        extracted_data.append({
            "original_filename": img.filename,
            "segmented_answers": student_segments,
            "validation": {
                "grading": grading,
                "teacher_reference_found": True,
                "teacher_filename": teacher_filename,
                "teacher_cached": teacher_from_cache,
                "teacher_segmentation_type": segmentation_type(teacher_segments),
                "student_segmentation_type": segmentation_type(student_segments),
            }
        })

    return {
        "student_id": student_id,
        "homework_id": homework_id,
        "title": student_rec.get("title"),
        "date": student_rec.get("date"),
        "completion_status": completion_status,

        "submission_remarks": submission_remarks,
        "teacher_image": teacher_filename,
        "teacher_url": teacher_url,
        "files_processed": len(images),
        "extracted_data": extracted_data,
        "message": "Teacher auto-fetched from ERP. Student images processed successfully.",
    }
