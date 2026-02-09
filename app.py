import os
import io
import re
import uuid
from datetime import datetime
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
# ✅ FALLBACK MESSAGES - Score-appropriate and varied
# =========================================================
HIGH_SCORE_MESSAGES = [
    "Excellent work! Your {:.0%} score shows great understanding of the material!",
    "Amazing job! You achieved {:.0%} - your hard work is paying off!",
    "Outstanding performance with {:.0%} - keep up the fantastic work!",
    "Brilliant! {:.0%} demonstrates excellent grasp of concepts!",
    "Perfect! {:.0%} shows you've mastered this topic completely!",
    "Wonderful! {:.0%} reflects your dedication and smart work!",
    "Spot on! {:.0%} shows you've understood everything perfectly!",
    "Spectacular! {:.0%} - you're doing an amazing job!",
    "Marvelous! {:.0%} shows exceptional understanding!",
    "Superb! {:.0%} - your efforts have truly paid off!",
    "Fantastic! {:.0%} demonstrates your excellent grasp!",
    "Incredible! {:.0%} - you're exceeding all expectations!",
    "Remarkable! {:.0%} shows true mastery of the subject!",
    "Brilliant work! {:.0%} reflects your hard work and talent!",
    "Spectacular! {:.0%} shows you're a natural!",
    "Outstanding! {:.0%} - you're crushing it!",
    "Magnificent! {:.0%} shows incredible dedication!",
    "Excellent! {:.0%} - you're making amazing progress!",
    "Superb work! {:.0%} shows your commitment!",
    "First class! {:.0%} - you're doing wonderfully!",
]

MEDIUM_SCORE_MESSAGES = [
    "Good effort! Your score of {:.0%} shows decent understanding. Review missed parts to improve!",
    "Solid work at {:.0%}. Focus on areas where you lost marks for next time!",
    "You're making progress at {:.0%}. Keep practicing the topics you missed!",
    "Nice try at {:.0%}. A bit more study will help you reach full marks!",
    "Nice work! {:.0%} shows potential - review and improve!",
    "Good progress! {:.0%} - keep pushing forward!",
    "Decent attempt at {:.0%}. Some areas need more attention!",
    "Good start at {:.0%}. Build on this foundation!",
    "Promising {:.0%}. Spend more time on challenging topics!",
    "Nearly there! {:.0%} - almost perfect, keep trying!",
    "Keep going! {:.0%} shows you're on the right track!",
    "Good improvement! {:.0%} - continue this positive trend!",
    "Nice effort! {:.0%} - review what you missed and grow!",
    "Well done! {:.0%} - a little more practice will help!",
    "Good job! {:.0%} - focus on weak areas next time!",
    "Promising! {:.0%} - you're getting closer to mastery!",
    "Keep studying! {:.0%} - every bit of effort counts!",
    "Nice work! {:.0%} - identify gaps and fill them!",
    "Growing! {:.0%} - you're making steady progress!",
    "Focused! {:.0%} - keep refining your understanding!",
]

LOW_SCORE_MESSAGES = [
    "Your score of {:.0%} shows you need to review the material more carefully.",
    "Keep trying! {:.0%} means there's room for improvement. Review the teacher's answers!",
    "Your submission scored {:.0%}. Please review the correct answers and try again!",
    "At {:.0%}, you'll need to study the material more thoroughly before resubmitting.",
    "{:.0%} suggests more practice is needed. Go through the concepts again!",
    "{:.0%} is a starting point. Focus on understanding the basics!",
    "{:.0%} indicates you should revisit the topics covered. Don't give up!",
    "{:.0%} means it's time for extra study. Review and try again!",
    "{:.0%} - please review the lesson materials and resubmit!",
    "{:.0%} shows you need more practice. Keep working at it!",
    "{:.0%} - every expert was once a beginner. Keep learning!",
    "{:.0%} - identify what you missed and study those areas!",
    "{:.0%} - review the reference materials carefully!",
    "{:.0%} - don't be discouraged, persistence pays off!",
    "{:.0%} - take time to understand each concept step by step!",
    "{:.0%} - practice makes perfect. Try again soon!",
    "{:.0%} - this is an opportunity to learn and grow!",
    "{:.0%} - focus on understanding, not just memorizing!",
    "{:.0%} - put in more time and effort to improve!",
    "{:.0%} - review, practice, and you'll get better!",
]


# =========================================================
# ✅ LLM REMARK (for individual image evaluation)
# =========================================================
def generate_llm_remark(
    teacher_text: str, 
    student_text: str, 
    sim_score: float, 
    threshold: float,
    completion_status: str = "N",
    unique_seed: str = ""
) -> str:
    """
    Generate AI-generated remark using OpenAI API for individual image evaluation.
    unique_seed ensures different outputs even for identical inputs.
    """
    if client is None:
        return "AI remark generation unavailable (OpenAI API key not configured)."

    # Keep excerpts reasonable
    teacher_excerpt = (teacher_text or "")[:800]
    student_excerpt = (student_text or "")[:800]
    
    # Determine if individual answer passed
    passed = sim_score >= threshold
    
    # System prompt for maximum variation
    system_prompt = (
        "You are a creative teacher giving unique feedback each time. "
        "CRITICAL: You MUST create COMPLETELY DIFFERENT responses for each submission. "
        "Never repeat the same words, phrases, or structure. "
        "Use different metaphors, emojis, encouragement styles, and expressions. "
        "Keep it concise but always fresh and unique."
    )
    
    # User prompt with unique_seed
    user_prompt = (
        f"SEED: {unique_seed} - USE THIS TO CREATE A UNIQUE RESPONSE\n\n"
        f"Teacher's answer:\n{teacher_excerpt}\n\n"
        f"Student's answer:\n{student_excerpt}\n\n"
        f"Score: {sim_score:.0%} (need {threshold:.0%} to pass)\n"
        f"Result: {'🎉 PERFECT!' if passed else '📚 Keep learning!'}\n\n"
        "Create a unique, different response every time. "
        "Use different words, emojis, and encouragement style than any previous response."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=80,
            temperature=2.0,  # Maximum randomness
        )
        remark = (resp.choices[0].message.content or "").strip()
        return remark if remark else "🌟 Great effort! Keep learning!"
    except Exception as e:
        print(f"OpenAI API error for individual remark: {e}")
        return "Your work has been submitted for review."


# =========================================================
# ✅ LLM SUBMISSION REMARK (overall submission feedback)
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
    submission_date: str,
    unique_seed: str = ""
) -> str:
    """
    Generate overall submission remarks using OpenAI API.
    unique_seed ensures different outputs even for identical inputs.
    """
    if client is None:
        return "AI feedback unavailable."

    if not student_texts:
        return "No submission found."

    teacher_excerpt = (teacher_text or "")[:800]
    
    num_images = len(student_texts)
    if scores and num_images > 0:
        avg_score = sum(scores) / num_images
        passed_count = sum(1 for score in scores if score >= threshold)
        pass_rate = (passed_count / num_images * 100)
    else:
        avg_score = 0
        passed_count = 0
        pass_rate = 0
    
    student_samples = []
    for i, (text, score) in enumerate(zip(student_texts, scores), 1):
        text_excerpt = (text or "")[:80].strip()
        if text_excerpt:
            pct = int(score * 100)
            student_samples.append(f"Part {i}: {pct}% match")
    
    # System prompt - MAXIMUM UNIQUENESS
    system_prompt = (
        "You are a creative feedback assistant. CRITICAL TASK: "
        "Generate a COMPLETELY UNIQUE feedback message every single time. "
        "NEVER repeat words, phrases, sentence structures, or feedback patterns. "
        "Use different emojis, metaphors, encouragement styles, and expressions. "
        "If you gave feedback before, make this one TOTALLY DIFFERENT. "
        "Maximum creativity required!"
    )
    
    # User prompt - FORCE variation
    user_prompt = (
        f"🌟 UNIQUE SEED: {unique_seed} - THIS MAKES EVERY RESPONSE DIFFERENT 🌟\n\n"
        f"Homework: {homework_title or 'Assignment'} | Student: {student_id}\n"
        f"Teacher's correct answer (excerpt):\n{teacher_excerpt[:500]}\n\n"
        f"📊 RESULTS:\n"
        f"• Average match: {avg_score:.0%} (threshold: {threshold:.0%})\n"
        f"• Parts passed: {passed_count}/{num_images}\n"
        f"• Status: {'✅ COMPLETE!' if completion_status == 'Y' else '📝 IN PROGRESS'}\n\n"
    )
    
    if student_samples:
        user_prompt += "📋 Details: " + " | ".join(student_samples) + "\n\n"
    
    user_prompt += (
        "🎯 YOUR TASK: Give unique, creative feedback that is DIFFERENT from any previous response. "
        "Use new words, different emojis, varied encouragement style. "
        "Make each submission feel special and one-of-a-kind!"
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=120,
            temperature=2.0,  # Maximum randomness for unique responses
        )
        remark = (resp.choices[0].message.content or "").strip()
        if remark:
            return remark
    except Exception as e:
        print(f"OpenAI error: {e}")
    
    # Score-appropriate fallback messages (20 options per category)
    if avg_score >= 0.8:
        fallbacks = HIGH_SCORE_MESSAGES
    elif avg_score >= 0.5:
        fallbacks = MEDIUM_SCORE_MESSAGES
    else:
        fallbacks = LOW_SCORE_MESSAGES
    
    # Select message based on unique_seed hash for consistency
    import random
    selected_index = hash(unique_seed) % len(fallbacks)
    return fallbacks[selected_index].format(avg_score)


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
    
    # Generate unique seeds for different remarks
    submission_seed = f"{datetime.now().isoformat()}_{uuid.uuid4().hex[:12]}"
    
    # Calculate completion_status
    calculated_completion_status = "Y" if scores and all(s >= threshold_f for s in scores) else "N"
    
    # Second pass: generate individual remarks
    for i, img in enumerate(images):
        grading = gradings[i]
        student_text = student_texts[i]
        score = grading.get("overall_score")
        
        # Generate unique seed for each image
        image_seed = f"{datetime.now().isoformat()}_{uuid.uuid4().hex[:12]}"
        
        if score is None:
            remark = "Unable to evaluate: reference or answer text is not readable."
        else:
            remark = generate_llm_remark(
                teacher_text, 
                student_text, 
                float(score), 
                threshold_f, 
                completion_status=calculated_completion_status,
                unique_seed=image_seed
            )

        remarks.append(remark)
        extracted_data.append({
            "original_filename": img.filename if hasattr(img, 'filename') else f"image_{i}.jpg",
            "student_text": student_text,
            "grading": grading,
            "ai_generated_remark": remark,
        })
    
    # Generate submission remarks
    submission_remarks = generate_llm_submission_remark(
        teacher_text=teacher_text,
        student_texts=student_texts,
        scores=scores,
        threshold=threshold_f,
        completion_status=calculated_completion_status,
        student_id=student_id,
        homework_id=homework_id,
        homework_title=student_rec.get("title", ""),
        submission_date=student_rec.get("date", ""),
        unique_seed=submission_seed
    )
    
    # Log the remark
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
        "submission_remarks": submission_remarks,
        "teacher_image": teacher_filename,
        "teacher_url": teacher_url,
        "files_processed": len(images),
        "extracted_data": extracted_data,
        "message": "All remarks generated with unique responses each time!",
    }
