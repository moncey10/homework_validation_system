import os
import io
import re
import json
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFilter
import pytesseract

from dotenv import load_dotenv
load_dotenv()

# Optional extractors for DOCX/PDF
try:
    from docx import Document  # python-docx
except Exception:
    Document = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from pdf2image import convert_from_bytes  # requires poppler
except Exception:
    convert_from_bytes = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ NEW Gemini SDK
try:
    from google import genai
except Exception as e:
    genai = None
    print(f"[WARN] google-genai import failed: {e}")


# =========================================================
# ✅ FASTAPI APP INSTANCE
# =========================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# ✅ ANSWER KEY (OPTIONAL)
# =========================================================
ANSWER_KEY_PATH = os.getenv("ANSWER_KEY_PATH", "answer_key.json")

def _load_answer_key() -> dict:
    try:
        with open(ANSWER_KEY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {ANSWER_KEY_PATH}: {e}")
        return {}

ANSWER_KEY = _load_answer_key()

def get_manual_reference_answer(homework_id: int) -> str:
    candidates = [f"hw{homework_id:02d}", f"hw{homework_id}"]
    for key in candidates:
        hw = ANSWER_KEY.get(key)
        if hw and isinstance(hw.get("questions"), list):
            answers = []
            for q in hw["questions"]:
                ans = (q.get("answer") or "").strip()
                if ans:
                    answers.append(ans)
            if answers:
                return "\n".join(answers)
    return ""


# =========================================================
# ✅ GOOGLE GEMINI CONFIG (AI STUDIO KEY)
# =========================================================
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
print(f"[DEBUG] GOOGLE_API_KEY = '{GOOGLE_API_KEY}'")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash").strip()

# Ensure correct prefix to avoid 404 model errors
if GEMINI_MODEL and not GEMINI_MODEL.startswith("models/"):
    GEMINI_MODEL = "models/" + GEMINI_MODEL

gemini_client = None
GEMINI_LAST_ERROR = ""

def _init_gemini_client():
    """Initialize Gemini client once per process."""
    global gemini_client, GEMINI_LAST_ERROR

    if gemini_client is not None:
        return

    if not genai:
        GEMINI_LAST_ERROR = "google-genai not installed / import failed"
        gemini_client = None
        return

    if not GOOGLE_API_KEY:
        GEMINI_LAST_ERROR = "GOOGLE_API_KEY not set"
        gemini_client = None
        return

    try:
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        GEMINI_LAST_ERROR = ""
        print("[INFO] Gemini client initialized")
    except Exception as e:
        gemini_client = None
        GEMINI_LAST_ERROR = str(e)
        print(f"[WARN] Gemini init failed: {GEMINI_LAST_ERROR}")

_init_gemini_client()

def parse_gemini_error(error_msg: str) -> dict:
    msg = (error_msg or "").strip()
    lower = msg.lower()

    if "service_disabled" in lower or "generativelanguage.googleapis.com" in lower:
        return {
            "ok": False,
            "error_type": "GEMINI_SERVICE_DISABLED",
            "message": msg,
            "fix_steps": [
                "Create a new API key in Google AI Studio",
                "Ensure the key is active and copied correctly into .env (GOOGLE_API_KEY=...)",
                "If still failing: open the Google Cloud project behind the key and enable Billing",
                "Enable Generative Language API (generativelanguage.googleapis.com) in that project",
                "Temporarily remove API key restrictions and test again",
            ],
        }

    if "api key" in lower or "invalid" in lower or "permission" in lower or "unauthorized" in lower:
        return {
            "ok": False,
            "error_type": "GEMINI_KEY_OR_PERMISSION_ERROR",
            "message": msg,
            "fix_steps": [
                "Regenerate / create a fresh key in Google AI Studio",
                "Update .env GOOGLE_API_KEY",
                "Restart uvicorn",
                "Do NOT print API key in logs",
            ],
        }

    return {"ok": False, "error_type": "GEMINI_ERROR", "message": msg}

def generate_gemini_response(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> str:
    """Return response text or empty string. Sets GEMINI_LAST_ERROR on failure."""
    global GEMINI_LAST_ERROR

    if gemini_client is None:
        if not GEMINI_LAST_ERROR:
            GEMINI_LAST_ERROR = "Gemini client not initialized"
        return ""

    try:
        contents = []
        if system_prompt:
            contents.append(system_prompt)
        contents.append(prompt)

        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = getattr(resp, "text", "") or ""
        text = text.strip()
        if text:
            GEMINI_LAST_ERROR = ""
        return text

    except Exception as e:
        GEMINI_LAST_ERROR = str(e)
        print(f"[ERROR] Gemini call failed: {GEMINI_LAST_ERROR}")
        return ""


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
# ✅ OCR + TEXT EXTRACTION
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
    if not image_bytes or len(image_bytes) < 50:
        raise HTTPException(status_code=400, detail=f"Invalid file: '{filename}' - empty/too small")

    valid_image_signatures = {
        b"\xff\xd8\xff": "JPEG",
        b"\x89PNG\r\n\x1a\n": "PNG",
        b"GIF87a": "GIF",
        b"GIF89a": "GIF",
        b"BM": "BMP",
    }

    is_valid = any(image_bytes.startswith(sig) for sig in valid_image_signatures)
    if not is_valid:
        head = image_bytes[:12]
        raise HTTPException(status_code=400, detail=f"Invalid image format: '{filename}' (header={head})")

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image '{filename}': {e}")

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

def _clean_extracted_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text_from_docx(docx_bytes: bytes, filename: str = "unknown.docx") -> str:
    if Document is None:
        raise HTTPException(status_code=500, detail="DOCX support not installed. Add 'python-docx'.")
    try:
        doc = Document(io.BytesIO(docx_bytes))
        parts = []
        for p in doc.paragraphs:
            if p.text and p.text.strip():
                parts.append(p.text.strip())
        for t in doc.tables:
            for row in t.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return _clean_extracted_text("\n".join(parts))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read DOCX '{filename}': {e}")

def extract_text_from_pdf(pdf_bytes: bytes, filename: str = "unknown.pdf") -> Dict[str, Any]:
    used_ocr = False
    extracted = ""

    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
            extracted = _clean_extracted_text("\n\n".join(parts))
        except Exception:
            extracted = ""

    if len(extracted) < 50:
        if convert_from_bytes is None:
            return {"text": extracted, "used_ocr": False, "needs_ocr": True}
        try:
            used_ocr = True
            pages = convert_from_bytes(pdf_bytes, dpi=250)
            page_texts = []
            for img in pages:
                img = _preprocess_for_ocr(img)
                t = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6") or ""
                if t.strip():
                    page_texts.append(t)
            extracted = _clean_extracted_text("\n\n".join(page_texts))
        except Exception as e:
            return {"text": extracted, "used_ocr": used_ocr, "needs_ocr": True, "ocr_error": str(e)}

    return {"text": extracted, "used_ocr": used_ocr, "needs_ocr": False}

async def extract_text_from_upload(file: UploadFile) -> Dict[str, Any]:
    filename = getattr(file, "filename", "") or "upload"
    content_type = (getattr(file, "content_type", "") or "").lower()
    data = await file.read()

    if not data or len(data) < 20:
        return {"text": "", "kind": "unknown", "used_ocr": False, "needs_ocr": False, "error": "empty"}

    ext = (os.path.splitext(filename)[1] or "").lower()

    is_image = content_type.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    is_pdf = (content_type == "application/pdf") or ext == ".pdf"
    is_docx = (content_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    }) or ext in {".docx", ".doc"}

    if is_image:
        try:
            return {"text": _clean_extracted_text(extract_text_from_image(data, filename=filename)), "kind": "image", "used_ocr": True, "needs_ocr": False}
        except HTTPException as e:
            return {"text": "", "kind": "image", "used_ocr": True, "needs_ocr": False, "error": e.detail}

    if is_docx:
        try:
            return {"text": _clean_extracted_text(extract_text_from_docx(data, filename=filename)), "kind": "docx", "used_ocr": False, "needs_ocr": False}
        except HTTPException as e:
            return {"text": "", "kind": "docx", "used_ocr": False, "needs_ocr": False, "error": e.detail}

    if is_pdf:
        info = extract_text_from_pdf(data, filename=filename)
        return {
            "text": info.get("text", ""),
            "kind": "pdf",
            "used_ocr": bool(info.get("used_ocr", False)),
            "needs_ocr": bool(info.get("needs_ocr", False)),
            "ocr_error": info.get("ocr_error"),
        }

    # fallback: try as image
    try:
        return {"text": _clean_extracted_text(extract_text_from_image(data, filename=filename)), "kind": "unknown_as_image", "used_ocr": True, "needs_ocr": False}
    except Exception:
        return {"text": "", "kind": "unknown", "used_ocr": False, "needs_ocr": False, "error": f"Unsupported file type: {content_type or ext or 'unknown'}"}


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


# =========================================================
# ✅ ROUTES
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/llm")
def health_llm():
    return {
        "ok": bool(gemini_client) and bool(GOOGLE_API_KEY),
        "gemini": {
            "sdk_import_ok": genai is not None,
            "configured": bool(GOOGLE_API_KEY),
            "client_ready": gemini_client is not None,
            "model": GEMINI_MODEL,
            "last_error": GEMINI_LAST_ERROR if GEMINI_LAST_ERROR else None,
        },
    }

@app.get("/gemini/models")
def gemini_models():
    if gemini_client is None:
        return {"ok": False, "error": "Gemini client not initialized", "last_error": GEMINI_LAST_ERROR}

    try:
        models = []
        for m in gemini_client.models.list():
            name = getattr(m, "name", "")
            supported = getattr(m, "supported_actions", None) or getattr(m, "supported_methods", None)
            models.append({"name": name, "supported": supported})
        return {"ok": True, "count": len(models), "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/homework/validate")
async def homework_validate(
    student_id: int = Form(...),
    homework_id: int = Form(...),
    sub_institute_id: int = Form(...),
    syear: str = Form(...),
    prompt: str = Form(...),
    teacher_file: UploadFile = File(...),
    student_file: UploadFile = File(...),
):
    teacher_info = await extract_text_from_upload(teacher_file)
    teacher_question_text = (teacher_info.get("text") or "").strip()

    student_info = await extract_text_from_upload(student_file)
    student_text = (student_info.get("text") or "").strip()

    MIN_WORDS = 8
    if len(student_text.split()) < MIN_WORDS:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "status": "Unreadable",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Answer text could not be read clearly. Please upload a clearer image/PDF/DOCX.",
            "teacher_extracted_text": teacher_question_text,
            "student_extracted_text": student_text,
            "extraction": {"teacher": {k: v for k, v in teacher_info.items() if k != "text"},
                           "student": {k: v for k, v in student_info.items() if k != "text"}},
            "llm_used": False,
        }

    if student_info.get("needs_ocr") and not student_text:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "status": "Unreadable",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "This PDF looks scanned. OCR is required (install pdf2image + poppler) or upload a clearer file.",
            "teacher_extracted_text": teacher_question_text,
            "student_extracted_text": student_text,
            "extraction": {"teacher": {k: v for k, v in teacher_info.items() if k != "text"},
                           "student": {k: v for k, v in student_info.items() if k != "text"}},
            "llm_used": False,
        }

    if not teacher_question_text and len((prompt or "").strip()) < 20:
        raise HTTPException(status_code=422, detail="Teacher question could not be extracted and prompt is too short. Provide a readable question file or include question text in prompt.")

    if gemini_client is None:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Gemini not configured. Check /health/llm.",
            "llm_used": False,
            "llm_error": parse_gemini_error(GEMINI_LAST_ERROR),
        }

    q_text = teacher_question_text[:2000]
    user_prompt = (
        f"{prompt.strip()}\n\n"
        f"QUESTION:\n{q_text}\n\n"
        'Return ONLY valid JSON with keys: {"ai_reference_answer": string, "key_points": [string, ...]}.'
    )

    response_text = generate_gemini_response(
        prompt=user_prompt,
        system_prompt="Generate a correct reference answer for homework evaluation. Output strict JSON only.",
        max_tokens=600,
        temperature=0.3,
    )

    if not response_text:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Gemini failed. Check llm_error and /health/llm.",
            "llm_used": False,
            "llm_error": parse_gemini_error(GEMINI_LAST_ERROR),
        }

    try:
        m = re.search(r"\{.*\}", response_text, flags=re.S)
        payload = json.loads(m.group(0) if m else response_text)
    except Exception as e:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Gemini returned non-JSON output. Fix prompt/LLM formatting.",
            "llm_used": False,
            "llm_error": {"ok": False, "error_type": "GEMINI_BAD_JSON", "message": str(e), "raw": response_text[:800]},
        }

    ai_reference_answer = (payload.get("ai_reference_answer") or "").strip()
    key_points = payload.get("key_points") or []
    if not isinstance(key_points, list):
        key_points = []
    key_points = [str(x).strip() for x in key_points if str(x).strip()]

    if not ai_reference_answer:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "AI returned empty reference answer.",
            "llm_used": True,
        }

    sim = cosine_sim(student_text, ai_reference_answer)

    student_low = student_text.lower()
    covered, missing = [], []
    for kp in key_points:
        if kp.lower() in student_low:
            covered.append(kp)
        else:
            missing.append(kp)

    coverage = (len(covered) / max(1, len(key_points))) if key_points else 0.0
    final = 0.6 * sim + 0.4 * coverage
    match_pct = int(round(final * 100))

    if match_pct >= 80:
        status = "Verified"
    elif match_pct >= 55:
        status = "Partial"
    else:
        status = "Needs Review"

    remark_prompt = (
        f"Match: {match_pct}%\n"
        f"Missing key points: {missing[:6]}\n\n"
        "Write a short, factual homework remark (2-4 lines). "
        "No marks. No overpraise. Mention 1-2 missing points if any."
    )

    resp2_prompt = (
        f"REFERENCE ANSWER:\n{ai_reference_answer[:900]}\n\n"
        f"STUDENT ANSWER:\n{student_text[:900]}\n\n"
        f"{remark_prompt}"
    )

    ai_generated_remark = None
    rule_based_remark = None
    remark_llm_used = False
    remark_llm_error = None

    ai_generated_remark = generate_gemini_response(
        prompt=resp2_prompt,
        system_prompt="You are a strict, helpful teacher. Be concise and factual.",
        max_tokens=120,
        temperature=0.6,
    )

    if ai_generated_remark:
        remark_llm_used = True
    else:
        remark_llm_error = GEMINI_LAST_ERROR or "Unknown LLM error"
        if status == "Verified":
            rule_based_remark = "Homework matches the expected answer well. Good coverage of the key ideas."
        elif status == "Partial":
            rule_based_remark = "Homework is partially correct. Improve coverage of missing key points and make the explanation clearer."
        else:
            rule_based_remark = "Homework does not match the expected answer enough. Please review the topic and resubmit with clearer, complete points."

    return {
        "student_id": student_id,
        "homework_id": homework_id,
        "sub_institute_id": sub_institute_id,
        "syear": syear,
        "status": status,
        "match_percentage": match_pct,
        "ai_generated_remark": ai_generated_remark,
        "rule_based_remark": rule_based_remark,
        "llm_used": True,
        "remark_llm_used": remark_llm_used,
        "remark_llm_error": remark_llm_error,
        "teacher_extracted_text": teacher_question_text,
        "student_extracted_text": student_text,
        "ai_reference_answer": ai_reference_answer,
        "key_points": key_points,
        "key_points_covered": covered,
        "key_points_missing": missing,
        "debug": {"similarity": sim, "coverage": coverage},
        "extraction": {"teacher": {k: v for k, v in teacher_info.items() if k != "text"},
                       "student": {k: v for k, v in student_info.items() if k != "text"}},
    }


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
    for i, img in enumerate(images, 1):
        student_bytes = await img.read()
        fname = (img.filename or f"image_{i}.jpg")
        student_text = extract_text_from_image(student_bytes, filename=fname)
        student_texts.append(student_text)

        grading = {
            "status": "EVALUATED" if teacher_text.strip() and student_text.strip() else "NO_TEXT",
            "overall_score": cosine_sim(student_text, teacher_text) if teacher_text.strip() and student_text.strip() else None,
            "threshold": threshold_f
        }
        score = grading.get("overall_score")
        gradings.append(grading)
        if score is not None:
            scores.append(float(score))

    calculated_completion_status = "Y" if scores and all(s >= threshold_f for s in scores) else "N"

    # Second pass: deterministic remarks
    for i, img in enumerate(images, 1):
        grading = gradings[i - 1]
        student_text = student_texts[i - 1]
        score = grading.get("overall_score")

        if score is None:
            remark = "Unable to evaluate: reference or answer text is not readable."
        else:
            pct = int(float(score) * 100)
            remark = f"Match {pct}%. " + ("Good work." if pct >= int(threshold_f * 100) else "Needs improvement. Review missing points and rewrite clearly.")

        remarks.append(remark)
        extracted_data.append({
            "original_filename": img.filename or f"image_{i}.jpg",
            "student_text": student_text,
            "grading": grading,
            "ai_generated_remark": remark,
        })

    return {
        "student_id": student_id,
        "homework_id": homework_id,
        "title": student_rec.get("title"),
        "date": student_rec.get("date"),
        "completion_status": student_rec.get("completion_status"),
        "calculated_completion_status": calculated_completion_status,
        "teacher_image": teacher_filename,
        "teacher_url": teacher_url,
        "files_processed": len(images),
        "extracted_data": extracted_data,
        "message": "Processed successfully",
    }
