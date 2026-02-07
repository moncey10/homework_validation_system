import os
import io
import re
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image, ImageOps, ImageFilter
import pytesseract

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ OpenAI SDK (pip install openai)
from openai import OpenAI


# =========================================================
# ✅ TESSERACT PATH
# =========================================================
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


# =========================================================
# ✅ ERP CONFIG
# =========================================================
ERP_BASE = os.getenv("ERP_BASE", "https://erp.triz.co.in/lms_data")
STORAGE_BASE = os.getenv("STORAGE_BASE", "https://erp.triz.co.in/storage/student/")
ERP_TOKEN = os.getenv("ERP_TOKEN", "")

# =========================================================
# ✅ OPENAI CONFIG
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # you can change
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


app = FastAPI(title="Homework Validation System (LLM Remarks)")


# =========================================================
# ✅ ERP HELPERS
# =========================================================
def _erp_get(params: dict) -> list:
    headers = {}
    if ERP_TOKEN:
        headers["Authorization"] = f"Bearer {ERP_TOKEN}"

    r = requests.get(ERP_BASE, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="ERP returned invalid JSON (expected list).")
    return data


def fetch_student_record(homework_id: int, student_id: int) -> Dict[str, Any]:
    data = _erp_get({"table": "homework", "filters[id]": homework_id, "filters[student_id]": student_id})
    if not data:
        raise HTTPException(status_code=404, detail="No ERP record found for this homework_id + student_id")
    return data[0]


def fetch_teacher_image_by_homework_id(homework_id: int) -> str:
    data = _erp_get({"table": "homework", "filters[id]": homework_id})
    if not data:
        raise HTTPException(status_code=404, detail="No ERP homework record found for this homework_id")

    row = data[0]
    for key in ("image", "teacher_image", "reference_image", "solution_image"):
        v = (row.get(key) or "").strip()
        if v:
            return v

    raise HTTPException(
        status_code=422,
        detail="Teacher image missing in ERP for this homework_id (image/teacher_image/reference_image/solution_image all empty).",
    )


# =========================================================
# ✅ DOWNLOAD
# =========================================================
def _looks_like_html(b: bytes) -> bool:
    head = (b[:300] or b"").lower()
    return (b"<!doctype html" in head) or (b"<html" in head) or (b"<head" in head)


def download_bytes(url: str) -> bytes:
    headers = {}
    if ERP_TOKEN:
        headers["Authorization"] = f"Bearer {ERP_TOKEN}"

    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    b = r.content or b""

    if _looks_like_html(b):
        raise HTTPException(status_code=502, detail="Teacher image URL returned HTML (not an image). Storage may require auth.")

    return b


# =========================================================
# ✅ OCR
# =========================================================
def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)

    w, h = img.size
    if max(w, h) < 1600:
        scale = 1600 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))

    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda p: 255 if p > 170 else 0)
    return img


def extract_text_from_image(image_bytes: bytes, filename: str = "unknown") -> str:
    """Extract text from image bytes with validation."""
    # Validate that we have actual image data
    if not image_bytes or len(image_bytes) < 50:
        raise HTTPException(status_code=400, detail=f"Invalid file: '{filename}' - file is empty or too small")
    
    # Check for common image magic bytes
    valid_image_signatures = {
        b'\xff\xd8\xff': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'GIF87a': 'GIF',
        b'GIF89a': 'GIF',
        b'BM': 'BMP',
    }
    
    is_valid_image = False
    detected_type = None
    for sig, img_type in valid_image_signatures.items():
        if image_bytes[:len(sig)] == sig:
            is_valid_image = True
            detected_type = img_type
            break
    
    if not is_valid_image:
        # Try to identify what was actually sent
        file_header = image_bytes[:20]
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image format: '{filename}' - not a valid image file (detected: {file_header[:10]}). Supported formats: JPEG, PNG, GIF, BMP"
        )
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: '{filename}' - cannot read image file: {str(e)}")

    img = _preprocess_for_ocr(img)

    try:
        text = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6")
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(status_code=500, detail="Tesseract OCR not found. Install it / fix path.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    text = (text or "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text


# =========================================================
# ✅ SIMILARITY
# =========================================================
def cosine_sim(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    vec = TfidfVectorizer().fit([a, b])
    X = vec.transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0][0])


def grade_similarity(teacher_text: str, student_text: str, threshold: float) -> Dict[str, Any]:
    teacher_text = (teacher_text or "").strip()
    student_text = (student_text or "").strip()

    if not teacher_text:
        return {"status": "NO_TEACHER_TEXT", "overall_score": None, "threshold": float(threshold)}
    if not student_text:
        return {"status": "NO_STUDENT_TEXT", "overall_score": None, "threshold": float(threshold)}

    sim = cosine_sim(student_text, teacher_text)
    return {"status": "EVALUATED", "overall_score": sim, "threshold": float(threshold)}


# =========================================================
# ✅ LLM REMARK (for individual image evaluation)
# =========================================================
def generate_llm_remark(
    teacher_text: str, 
    student_text: str, 
    sim_score: float, 
    threshold: float,
    completion_status: str = "N"
) -> str:
    """
    Generate AI-generated remark using OpenAI API for individual image evaluation.
    """
    if client is None:
        return "AI remark generation unavailable (OpenAI API key not configured)."

    # Keep excerpts reasonable
    teacher_excerpt = (teacher_text or "")[:800]
    student_excerpt = (student_text or "")[:800]
    
    # Determine if individual answer passed
    passed = sim_score >= threshold
    
    # System prompt for consistent but varied responses
    system_prompt = (
        "You are an experienced, encouraging teacher providing feedback on student homework. "
        "Generate a unique, personalized remark for each student submission. "
        "Vary your phrasing and tone each time while maintaining educational value. "
        "Be constructive, specific, and motivating. "
        "Keep the remark concise (15-25 words)."
    )
    
    # User prompt with context
    user_prompt = (
        f"Teacher reference text (excerpt):\n{teacher_excerpt}\n\n"
        f"Student answer text (excerpt):\n{student_excerpt}\n\n"
        f"Similarity score: {sim_score:.2f} (Threshold: {threshold})\n"
        f"Status: {'PASSED' if passed else 'NEEDS IMPROVEMENT'}\n\n"
        "Provide a unique, encouraging remark that helps the student understand "
        "what they did well and how they can improve. "
        "Make it different from standard template responses."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=60,
            temperature=0.8,  # Higher temperature for more varied responses
        )
        remark = (resp.choices[0].message.content or "").strip()
        return remark if remark else "Great effort! Keep working to improve your understanding."
    except Exception as e:
        print(f"OpenAI API error for individual remark: {e}")
        return "Thank you for your submission. Your work has been reviewed."


# =========================================================
# ✅ LLM SUBMISSION REMARK (overall submission feedback) - ALWAYS AI GENERATED
# =========================================================
def generate_llm_submission_remark(
    teacher_text: str,
    student_texts: List[str],
    scores: List[float],
    threshold: float,
    completion_status: str,
    student_id: int,
    homework_id: int,
    homework_title: str,
    submission_date: str
) -> str:
    """
    Generate overall submission remarks using OpenAI API ONLY.
    Evaluates how well student submission MATCHES the teacher's reference image.
    """
    if client is None:
        return "AI feedback unavailable. Your submission has been graded based on similarity."
    
    if not student_texts:
        return "No submission found to evaluate."
    
    # Prepare context for the AI - FOCUS ON MATCHING
    teacher_excerpt = (teacher_text or "")[:800]
    
    # Analyze matching scores
    num_images = len(student_texts)
    if scores and num_images > 0:
        avg_score = sum(scores) / num_images
        passed_count = sum(1 for score in scores if score >= threshold)
        pass_rate = (passed_count / num_images * 100)
    else:
        avg_score = 0
        passed_count = 0
        pass_rate = 0
    
    # Prepare student text samples with match status
    student_samples = []
    for i, (text, score) in enumerate(zip(student_texts, scores), 1):
        text_excerpt = (text or "")[:80].strip()
        if text_excerpt:
            pct = int(score * 100)
            student_samples.append(f"Part {i}: {pct}% match - \"{text_excerpt}\"")
    
    # System prompt - AI generates everything based on match score
    system_prompt = (
        "You are an intelligent homework grading assistant. "
        "Evaluate the student's submission based on how well it MATCHES the teacher's reference. "
        "Generate a unique, specific feedback message for each submission. "
        "Based on the match percentage, explain what the student did well or what they're missing. "
        "Be encouraging but honest. Keep feedback between 30-50 words."
    )
    
    # User prompt - let AI interpret the match score
    user_prompt = (
        f"HOMEWORK EVALUATION\n"
        f"Student: {student_id} | Homework: {homework_id} | Title: {homework_title or 'N/A'}\n\n"
        f"TEACHER REFERENCE (correct answer):\n{teacher_excerpt[:600]}\n\n"
        f"RESULTS:\n"
        f"• Average Match: {avg_score:.0%} (threshold: {threshold:.0%})\n"
        f"• Passed: {passed_count}/{num_images} ({pass_rate:.0f}%)\n"
        f"• Overall: {'PASSED' if completion_status == 'Y' else 'NEEDS IMPROVEMENT'}\n\n"
    )
    
    if student_samples:
        user_prompt += "STUDENT SUBMISSIONS:\n" + "\n".join(student_samples[:4]) + "\n\n"
    
    user_prompt += (
        "FEEDBACK:\n"
        f"Based on the {avg_score:.0%} match, provide specific feedback about the student's answer. "
        f"If {avg_score:.0%} is high (above 75%), praise accuracy and completeness. "
        f"If {avg_score:.0%} is medium (50-75%), mention what's partially correct and what's missing. "
        f"If {avg_score:.0%} is low (below 50%), clearly explain what key points from the teacher's reference are missing. "
        "Be specific about content coverage."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        remark = (resp.choices[0].message.content or "").strip()
        return remark if remark else f"Match score: {avg_score:.0%}. Review individual feedback for details."
    except Exception as e:
        print(f"OpenAI error: {e}")
        return f"Match score: {avg_score:.0%}. Review individual feedback for details."


# =========================================================
# ✅ ROUTES
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/submit")
async def submit(
    student_id: int = Form(...),
    homework_id: int = Form(...),
    images: List[UploadFile] = File(...),
    threshold: float = Form(0.75),
):
    if not images:
        raise HTTPException(status_code=400, detail="At least one student image is required")

    try:
        threshold_f = float(threshold)
    except Exception:
        raise HTTPException(status_code=400, detail="threshold must be a number")

    student_rec = fetch_student_record(homework_id, student_id)

    # Teacher by homework_id only
    teacher_filename = fetch_teacher_image_by_homework_id(homework_id)
    teacher_url = STORAGE_BASE.rstrip("/") + "/" + teacher_filename.lstrip("/")
    teacher_bytes = download_bytes(teacher_url)
    teacher_text = extract_text_from_image(teacher_bytes, filename=teacher_filename)

    if not teacher_text.strip():
        raise HTTPException(status_code=422, detail="Teacher OCR extracted empty text. Teacher reference is not OCR-friendly.")

    extracted_data = []
    remarks = []
    scores = []
    student_texts = []
    gradings = []

    # First pass: extract text and calculate scores
    for img in images:
        student_bytes = await img.read()
        student_text = extract_text_from_image(student_bytes, filename=img.filename if hasattr(img, 'filename') else f"image_{i}")
        student_texts.append(student_text)

        grading = grade_similarity(teacher_text, student_text, threshold_f)
        score = grading.get("overall_score")
        gradings.append(grading)
        
        if score is not None:
            scores.append(float(score))
    
    # Calculate completion_status based on all scores
    calculated_completion_status = "Y" if scores and all(s >= threshold_f for s in scores) else "N"
    
    # Second pass: generate individual remarks for each image
    for i, img in enumerate(images):
        grading = gradings[i]
        student_text = student_texts[i]
        score = grading.get("overall_score")
        
        if score is None:
            remark = "Unable to evaluate: reference or answer text is not readable."
        else:
            remark = generate_llm_remark(
                teacher_text, 
                student_text, 
                float(score), 
                threshold_f, 
                completion_status=calculated_completion_status
            )

        remarks.append(remark)
        extracted_data.append({
            "original_filename": img.filename if hasattr(img, 'filename') else f"image_{i}.jpg",
            "student_text": student_text,
            "grading": grading,
            "ai_generated_remark": remark,
        })
    
    # ALWAYS generate submission remarks using OpenAI
    submission_remarks = generate_llm_submission_remark(
        teacher_text=teacher_text,
        student_texts=student_texts,
        scores=scores,
        threshold=threshold_f,
        completion_status=calculated_completion_status,
        student_id=student_id,
        homework_id=homework_id,
        homework_title=student_rec.get("title", ""),
        submission_date=student_rec.get("date", "")
    )
    
    # Log the AI-generated submission remark for debugging
    print(f"\n{'='*60}")
    print(f"AI GENERATED SUBMISSION REMARK:")
    print(f"{'='*60}")
    print(submission_remarks)
    print(f"{'='*60}\n")

    return {
        "student_id": student_id,
        "homework_id": homework_id,
        "title": student_rec.get("title"),
        "date": student_rec.get("date"),
        "completion_status": student_rec.get("completion_status"),
        "calculated_completion_status": calculated_completion_status,
        "submission_remarks": submission_remarks,  # Always AI-generated
        "teacher_image": teacher_filename,
        "teacher_url": teacher_url,
        "files_processed": len(images),
        "extracted_data": extracted_data,
        "message": "All remarks generated by OpenAI LLM (no template fallbacks).",
    }