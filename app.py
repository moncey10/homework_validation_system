
# app.py
import os
import io
import re
import json
from typing import Dict, Any, List, Tuple

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


# ✅ Gemini SDK
try:
    from google import genai
except Exception as e:
    genai = None
    print(f"[WARN] google-genai import failed: {e}")

# ✅ Google Cloud Vision SDK (for better handwritten OCR)
try:
    from google.cloud import vision
    from google.cloud.vision_v1 import types
    google_vision_available = True
except Exception as e:
    google_vision_available = False
    print(f"[WARN] google-cloud-vision import failed: {e}")



app = FastAPI()

import os

@app.get("/debug/env")
def debug_env():
    return {
        "has_gemini_key": bool(os.getenv("GEMINI_API_KEY")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"



ERP_BASE = os.getenv("ERP_BASE", "https://erp.triz.co.in/lms_data")
STORAGE_BASE = os.getenv("STORAGE_BASE", "https://erp.triz.co.in/storage/student/")
ERP_TOKEN = os.getenv("ERP_TOKEN", "")



GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_MODEL = (os.getenv("GEMINI_MODEL", "models/gemini-flash-lite-latest") or "").strip()
if GEMINI_MODEL and not GEMINI_MODEL.startswith("models/"):
    GEMINI_MODEL = "models/" + GEMINI_MODEL


GOOGLE_CLOUD_VISION_API_KEY = (os.getenv("GCV_API_KEY") or "").strip()
# Fall back to Gemini API key if no separate Vision key provided
if not GOOGLE_CLOUD_VISION_API_KEY and GOOGLE_API_KEY:
    GOOGLE_CLOUD_VISION_API_KEY = GOOGLE_API_KEY

vision_client = None
if google_vision_available and GOOGLE_CLOUD_VISION_API_KEY:
    try:
        # Use API key authentication
        vision_client = vision.ImageAnnotatorClient(client_options={
            'api_key': GOOGLE_CLOUD_VISION_API_KEY
        })
        print("[INFO] Google Cloud Vision client initialized")
    except Exception as e:
        print(f"[WARN] Google Cloud Vision init failed: {e}")

gemini_client = None
GEMINI_LAST_ERROR = ""


def _init_gemini_client() -> None:
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
        return {"ok": False, "error_type": "GEMINI_SERVICE_DISABLED", "message": msg}

    if "api key" in lower or "invalid" in lower or "permission" in lower or "unauthorized" in lower:
        return {"ok": False, "error_type": "GEMINI_KEY_OR_PERMISSION_ERROR", "message": msg}

    return {"ok": False, "error_type": "GEMINI_ERROR", "message": msg}


def generate_gemini_response(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 650,
    temperature: float = 0.3,
) -> str:
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
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        text = (getattr(resp, "text", "") or "").strip()
        if text:
            GEMINI_LAST_ERROR = ""
        return text
    except Exception as e:
        GEMINI_LAST_ERROR = str(e)
        print(f"[ERROR] Gemini call failed: {GEMINI_LAST_ERROR}")
        return ""

import time

def generate_gemini_with_retry(prompt: str, system_prompt: str, max_tokens=450, temperature=0.3, retries=3) -> str:
    last = ""
    for i in range(retries):
        text = generate_gemini_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if text:
            return text
        last = GEMINI_LAST_ERROR
        # small backoff
        time.sleep(1 + i)
    return ""

def cheap_overlap_score(student_text: str, prompt: str) -> int:
    # remove tiny words
    def tokens(s):
        return {w for w in re.findall(r"[a-zA-Z]{4,}", (s or "").lower())}
    s = tokens(student_text)
    p = tokens(prompt)
    if not s or not p:
        return 0
    overlap = len(s & p) / max(1, len(p))
    # map to a sane range
    return int(round(min(0.6, overlap) * 100))  # cap at 60



def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def cosine_sim(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    vec = TfidfVectorizer().fit([a, b])
    X = vec.transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0][0])


def normalize_level(level: str) -> str:
    l = (level or "").strip().lower()
    if l in ("easy",):
        return "Easy"
    if l in ("hard",):
        return "Hard"
    if l in ("meadium", "mediam", "medium"):
        return "Medium"
    return "Medium"


def level_policy(student_level: str) -> dict:
    lvl = normalize_level(student_level).lower()
    if lvl == "easy":
        return {"w_sim": 0.8, "w_cov": 0.2, "verified": 65, "partial": 40, "kp_thr": 0.25}
    if lvl == "hard":
        return {"w_sim": 0.4, "w_cov": 0.6, "verified": 85, "partial": 65, "kp_thr": 0.40}
    return {"w_sim": 0.6, "w_cov": 0.4, "verified": 75, "partial": 55, "kp_thr": 0.20}


def mcq_partial_credit(student_level: str) -> dict:
    """
    Returns partial credit percentage for MCQ questions based on student level.
    This allows easier students to get partial marks even if they get some questions wrong.
    
    Returns dict with:
    - credit_per_question: percentage earned per correct answer
    - passing_threshold: minimum percentage needed to pass
    """
    lvl = normalize_level(student_level).lower()
    if lvl == "easy":
        # Easy students get 50% credit per correct answer
        return {"credit_per_question": 50, "passing_threshold": 50}
    if lvl == "hard":
        # Hard students need 100% - no partial credit
        return {"credit_per_question": 100, "passing_threshold": 100}
    # Medium students get 75% credit per correct answer
    return {"credit_per_question": 75, "passing_threshold": 75}


def keypoint_coverage(student_text: str, key_points: List[str], kp_threshold: float) -> Tuple[List[str], List[str], float]:
    covered, missing = [], []
    for kp in key_points:
        kp = (kp or "").strip()
        if not kp:
            continue
        s = cosine_sim(kp, student_text)
        if s >= kp_threshold:
            covered.append(kp)
        else:
            missing.append(kp)

    total = len(covered) + len(missing)
    coverage = (len(covered) / total) if total else 0.0
    return covered, missing, coverage



def infer_question_type_from_prompt(prompt: str) -> str:
    p = _norm(prompt)

    # Explicit markers - check for (mcq) first since it's common in parentheses
    if re.search(r"\(mcq\)", p) or re.search(r"\btype\s*:\s*mcq\b", p) or re.search(r"\bquestion_type\s*:\s*mcq\b", p):
        return "mcq"
    if re.search(r"\btype\s*:\s*narrative\b", p) or re.search(r"\bquestion_type\s*:\s*narrative\b", p):
        return "narrative"

    # Heuristic: options A/B/C/D exist -> likely MCQ
    if re.search(r"\b(a|b|c|d)\s*[\)\.]\s+", p) or "option a" in p or "option b" in p:
        return "mcq"

    return "narrative"


def parse_questions_from_prompt(prompt: str) -> List[Dict[str, Any]]:
    """
    Parse individual questions from the prompt, detecting MCQ vs Narrative for each.
    Returns list of dicts with: qid, type, question_text, correct_answer (for MCQ)
    """
    questions = []
    # Match patterns like "Q1:", "Q2.", "Question 1:", etc.
    q_pattern = re.compile(r'(Q\s*\d+[.:]\s*|Question\s*\d+[.:]\s*)(.*?)(?=(Q\s*\d|Question\s*\d|$))', re.IGNORECASE | re.DOTALL)
    
    # Alternative: split by Q1, Q2, etc.
    lines = prompt.split('\n')
    current_q = None
    current_type = None
    current_qid = None
    current_correct = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect new question
        q_match = re.match(r'^(Q\s*\d+|Question\s*\d+)[.:]\s*(.*)', line, re.IGNORECASE)
        if q_match:
            # Save previous question if exists
            if current_q is not None:
                questions.append({
                    'qid': current_qid,
                    'type': current_type,
                    'question': current_q,
                    'correct_answer': current_correct
                })
            # Start new question
            current_qid = q_match.group(1).strip()
            remaining = q_match.group(2).strip()
            current_q = remaining
            current_type = None
            current_correct = None
            
            # Check if this is MCQ or Narrative
            line_lower = line.lower()
            if '(mcq)' in line_lower or 'multiple choice' in line_lower or 'type: mcq' in line_lower:
                current_type = 'mcq'
            elif 'narrative' in line_lower or 'type: narrative' in line_lower:
                current_type = 'narrative'
        else:
            # This line belongs to current question
            if current_q is not None:
                current_q += ' ' + line
                
                # Check for type markers
                line_lower = line.lower()
                if current_type is None:
                    if '(mcq)' in line_lower or 'multiple choice' in line_lower or 'type: mcq' in line_lower:
                        current_type = 'mcq'
                    elif 'narrative' in line_lower or 'type: narrative' in line_lower:
                        current_type = 'narrative'
                
                # Check for correct answer (for MCQ)
                if current_type == 'mcq':
                    # First check: is this line "Correct Answer(s):" with nothing after it?
                    # If so, we need to look for the answer on the next line
                    if re.search(r'^correct\s*answer\s*\(?s\)?\s*[:\.]?\s*$', line, re.IGNORECASE):
                        # Set flag to look for answer on next line
                        current_q += ' [CORRECT_ANSWER_PENDING]'
                        continue
                    
                    # Check if we have a pending correct answer marker
                    if '[CORRECT_ANSWER_PENDING]' in current_q:
                        # This line should contain the answer like "A. Devdatta"
                        letter_match = re.search(r'^([A-D])\.?\s*', line)
                        if letter_match:
                            current_correct = letter_match.group(1).upper()
                            # Remove the pending marker from question text
                            current_q = current_q.replace(' [CORRECT_ANSWER_PENDING]', '')
                            continue
                        else:
                            # Not a letter, remove the pending marker
                            current_q = current_q.replace(' [CORRECT_ANSWER_PENDING]', '')
                    
                    # Look for "Correct Answer(s):" or "Correct:" or "Answer:" in same line
                    # Support formats: "Correct Answer(s): A.", "Correct: B", "Answer: C"
                    correct_match = re.search(r'(?:Correct\s*(?:Answer)?|Answer)[:.]\s*(?:[A-D]\.?\s*)?(.+)', line, re.IGNORECASE)
                    if correct_match and not current_correct:
                        # Extract just the letter (A, B, C, or D)
                        correct_text = correct_match.group(1).strip()
                        letter_match = re.search(r'^([A-D])\b', correct_text)
                        if letter_match:
                            current_correct = letter_match.group(1).upper()
                        else:
                            # Try to extract first letter
                            current_correct = correct_text[0].upper() if correct_text else None
    
    # Don't forget the last question
    if current_q is not None:
        questions.append({
            'qid': current_qid,
            'type': current_type,
            'question': current_q,
            'correct_answer': current_correct
        })
    
    # If no questions parsed, fall back to old behavior
    if not questions:
        qtype = infer_question_type_from_prompt(prompt)
        return [{'qid': 'Q1', 'type': qtype, 'question': prompt, 'correct_answer': None}]
    
    return questions


def extract_mcq_choice(text: str) -> str:
    """
    Extract chosen option from student text:
    supports: A, (B), Option C, Ans: D, Answer: B
    """
    t = _norm(text)

    m = re.search(r"\b(answer|ans|selected)\s*[:\-]?\s*\(?\s*([a-d])\s*\)?\b", t)
    if m:
        return m.group(2)

    m2 = re.search(r"\boption\s*([a-d])\b", t)
    if m2:
        return m2.group(1)

    m3 = re.search(r"^\(?\s*([a-d])\s*\)?$", t.strip())
    if m3:
        return m3.group(1)

    # last-resort: find first standalone A/B/C/D
    m4 = re.search(r"\b([a-d])\b", t)
    if m4:
        return m4.group(1)

    return ""


def extract_mcq_answers_with_qid(text: str) -> Dict[str, str]:
    """
    Extract MCQ answers WITH question numbers from student text.
    This handles shuffled answers where question numbers are needed to match.
    
    Supports patterns like:
    - "Q1: A, Q2: C, Q3: B"
    - "Q1. A Q2. C Q3. B"
    - "1) A 2) C 3) B"
    - "Answer 1: A Answer 2: C Answer 3: B"
    - "Q1 A Q2 C Q3 B" (space separated)
    
    Returns dict like: {"Q1": "A", "Q2": "C", "Q3": "B"}
    """
    results = {}
    t = (text or "").strip()
    
    if not t:
        return results

    # Pattern 1: Q1: A, Q2. B, Q3 - C, Question 4: D
    pattern1 = re.compile(r'(Q(?:uestion)?\s*(\d+))[:.\-\s]+([a-dA-D])', re.IGNORECASE)
    for match in pattern1.finditer(t):
        qnum = match.group(2)
        answer = match.group(3).upper()
        results[f"Q{qnum}"] = answer
    
    # Pattern 2: 1) A, 2) B, 3: C (numbered without Q prefix)
    pattern2 = re.compile(r'(?:^|\s)(\d+)\s*[\):\.]\s*([a-dA-D])(?:\s|$)', re.IGNORECASE)
    for match in pattern2.finditer(t):
        qnum = match.group(1)
        answer = match.group(2).upper()
        # Only add if not already found (Q pattern takes priority)
        if f"Q{qnum}" not in results:
            results[f"Q{qnum}"] = answer
    
    # Pattern 3: "Answer for Q1 is A", "Answer to question 2: B"
    pattern3 = re.compile(r'(?:answer|ans)\s*(?:for|to)?\s*(?:Q(?:uestion)?\s*)?(\d+)\s*(?:is|was)?\s*[:\-]?\s*([a-dA-D])', re.IGNORECASE)
    for match in pattern3.finditer(t):
        qnum = match.group(1)
        answer = match.group(2).upper()
        if f"Q{qnum}" not in results:
            results[f"Q{qnum}"] = answer
    
    # Pattern 4: Line by line format like "Q1 A" or "1 A" on same line
    pattern4 = re.compile(r'(?:^|\n)\s*(Q(?:uestion)?\s*)?(\d+)\s+([a-dA-D])\s*(?:\n|\s{2,}|$)', re.IGNORECASE)
    for match in pattern4.finditer(t):
        qnum = match.group(2)
        answer = match.group(3).upper()
        if f"Q{qnum}" not in results:
            results[f"Q{qnum}"] = answer
    
    return results


def extract_correct_mcq_from_prompt(prompt: str) -> str:
    """
    This is IMPORTANT:
    Your prompt must contain correct option somewhere like:
      - Correct: B
      - Answer: C
      - correct_option: D
      - Correct Answer(s): A. Devdatta
    or JSON: {"correct_option":"B"}
    
    Supports formats:
      - "Correct Answer: A"
      - "Correct Answer(s): A. Devdatta"
      - "Correct: B"
      - "Answer: C"
    """
    p = (prompt or "").strip()
    if not p:
        return ""

    # JSON prompt support
    if p.startswith("{") and p.endswith("}"):
        try:
            obj = json.loads(p)
            for k in ("correct_option", "correct", "answer", "ans"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return extract_mcq_choice(v)
        except Exception:
            pass

    # Text prompt support - new format: "Correct Answer(s): A. Devdatta" or "Correct Answer: B"
    t = _norm(p)
    
    # Pattern 1: "Correct Answer(s): A. ..." or "Correct Answer: B. ..."
    # This handles format like "Correct Answer(s): A. Devdatta" or "Correct Answer(s):
    #    A. Devdatta"
    m1 = re.search(r"correct\s*answer\s*\(?s\)?\s*[:\.]\s*([a-d])\.?\s*", t)
    if m1:
        return m1.group(1)
    
    # Pattern 1b: Handle multi-line format where answer is on next line like:
    # "Correct Answer(s):\n   A. Devdatta"
    m1b = re.search(r"correct\s*answer\s*\(?s\)?\s*[:\.]\s*\n\s*([a-d])\.?", t)
    if m1b:
        return m1b.group(1)
    
    # Pattern 1c: Handle format with option text after letter like "Correct Answer(s): A. Devdatta"
    m1c = re.search(r"correct\s*answer\s*\(?s\)?\s*[:\.]\s*([a-d])\.", t)
    if m1c:
        return m1c.group(1)
    
    # Pattern 2: "Correct: A" or "Answer: B" (original pattern)
    m = re.search(r"\b(correct|answer|ans)\s*[:\-]?\s*\(?\s*([a-d])\s*\)?\b", t)
    if m:
        return m.group(2)

    return ""



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


def fetch_student_level_from_erp(row: Dict[str, Any]) -> str:
    """
    ERP field name is not guaranteed; try common ones.
    """
    for k in ("student_level", "level", "difficulty", "difficulty_level"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return normalize_level(v)
    return "Medium"



def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Enhanced preprocessing for better OCR on handwritten images.
    Includes adaptive thresholding, noise removal, and contrast enhancement.
    """
    # Convert to grayscale
    img = img.convert("L")
    
    w, h = img.size
    
    # Scale up for better detail (especially for handwritten)
    if max(w, h) < 2000:
        scale = 2000 / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Apply adaptive thresholding for better handwritten recognition
    from PIL import ImageFilter
    
    # Try multiple preprocessing approaches and use the best
    img_enhanced = img
    
    # Method 1: Increase contrast significantly
    img_contrast = img.point(lambda p: 255 if p > 180 else int(p * 1.5))
    
    # Method 2: Apply sharpening twice for handwritten
    img_sharp = img.filter(ImageFilter.SHARPEN)
    img_sharp = img_sharp.filter(ImageFilter.SHARPEN)
    
    # Method 3: Apply unsharp mask for edge enhancement
    img_unsharp = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # Use the sharpened version as primary
    img = img_sharp
    
    # Apply binary threshold with lower cutoff to capture lighter handwriting
    img = img.point(lambda p: 255 if p > 160 else 0)
    
    return img


def _extract_text_google_vision(image_bytes: bytes) -> str:
    """
    Extract text using Google Cloud Vision API - much better for handwriting.
    Returns empty string if API is not available.
    """
    global vision_client
    
    if not vision_client:
        return ""
    
    try:
        # Create image object
        image = vision.Image(content=image_bytes)
        
        # Use document text detection for better handwriting
        response = vision_client.document_text_detection(image=image)
        
        if response.texts:
            return "\n".join([t.description for t in response.texts])
        return ""
    except Exception as e:
        print(f"[WARN] Google Vision OCR failed: {e}")
        return ""


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

    # First try Google Cloud Vision (better for handwriting)
    if vision_client:
        gv_text = _extract_text_google_vision(image_bytes)
        if gv_text and len(gv_text.strip()) > 10:
            return _clean_extracted_text(gv_text)
    
    # Fallback to Tesseract with improved preprocessing
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image '{filename}': {e}")

    img = _preprocess_for_ocr(img)

    # Try multiple OCR configurations for better handwritten recognition
    ocr_configs = [
        "--oem 3 --psm 6",  # Default
        "--oem 3 --psm 4",  # Treat as single column
        "--oem 1 --psm 3",  # Fully automatic
    ]
    
    best_text = ""
    best_confidence = 0
    
    for config in ocr_configs:
        try:
            text = pytesseract.image_to_string(img, lang="eng", config=config)
            if text and len(text.strip()) > len(best_text.strip()):
                best_text = text
        except Exception:
            continue
    
    if not best_text:
        # Fallback to default if all fail
        try:
            best_text = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6")
        except pytesseract.TesseractNotFoundError:
            raise HTTPException(status_code=500, detail="Tesseract OCR not found. Install it / fix path.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    text = (best_text or "").strip()
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
            # Higher DPI for better handwritten OCR
            pages = convert_from_bytes(pdf_bytes, dpi=300)
            page_texts = []
            for img in pages:
                # Use the improved preprocessing
                img = _preprocess_for_ocr(img)
                
                # Try multiple OCR configs
                for config in ["--oem 3 --psm 6", "--oem 3 --psm 4", "--oem 1 --psm 3"]:
                    try:
                        t = pytesseract.image_to_string(img, lang="eng", config=config) or ""
                        if t.strip() and len(t.strip()) > 20:
                            page_texts.append(t)
                            break
                    except:
                        continue
            
            if page_texts:
                extracted = _clean_extracted_text("\n\n".join(page_texts))
            else:
                # Final fallback with default config
                img = pages[0] if pages else None
                if img:
                    img = _preprocess_for_ocr(img)
                    extracted = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6") or ""
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
            return {"text": _clean_extracted_text(extract_text_from_image(data, filename=filename)),
                    "kind": "image", "used_ocr": True, "needs_ocr": False}
        except HTTPException as e:
            return {"text": "", "kind": "image", "used_ocr": True, "needs_ocr": False, "error": e.detail}

    if is_docx:
        try:
            return {"text": _clean_extracted_text(extract_text_from_docx(data, filename=filename)),
                    "kind": "docx", "used_ocr": False, "needs_ocr": False}
        except HTTPException as e:
            return {"text": "", "kind": "docx", "used_ocr": False, "needs_ocr": False, "error": e.detail}

    if is_pdf:
        info = extract_text_from_pdf(data, filename=filename)
        return {"text": info.get("text", ""), "kind": "pdf",
                "used_ocr": bool(info.get("used_ocr", False)),
                "needs_ocr": bool(info.get("needs_ocr", False)),
                "ocr_error": info.get("ocr_error")}

    # fallback: try as image
    try:
        return {"text": _clean_extracted_text(extract_text_from_image(data, filename=filename)),
                "kind": "unknown_as_image", "used_ocr": True, "needs_ocr": False}
    except Exception:
        return {"text": "", "kind": "unknown", "used_ocr": False, "needs_ocr": False,
                "error": f"Unsupported file type: {content_type or ext or 'unknown'}"}





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


@app.post("/homework/validate")
async def homework_validate(
    student_id: int = Form(...),
    homework_id: int = Form(...),
    sub_institute_id: int = Form(...),
    syear: str = Form(...),
    prompt: str = Form(...),
    student_file: UploadFile = File(...),
):
    # 0) Fetch ERP record -> get student_level automatically
    erp_row = fetch_student_record(homework_id, student_id)
    student_level = fetch_student_level_from_erp(erp_row)
    policy = level_policy(student_level)

    # 1) Infer question_type from prompt automatically (NO EXTRA FIELD)
    # Try to parse mixed questions first
    parsed_questions = parse_questions_from_prompt(prompt)
    has_mcq = any(q.get('type') == 'mcq' for q in parsed_questions)
    has_narrative = any(q.get('type') == 'narrative' for q in parsed_questions)
    
    # Determine overall question type for backwards compatibility
    if has_mcq and has_narrative:
        question_type = "mixed"
    elif has_mcq:
        question_type = "mcq"
    elif has_narrative:
        question_type = "narrative"
    else:
        question_type = infer_question_type_from_prompt(prompt)

    # 2) Extract student text
    student_info = await extract_text_from_upload(student_file)
    student_text = (student_info.get("text") or "").strip()

    MIN_WORDS = 3 if question_type == "mcq" else 8
    if len(student_text.split()) < MIN_WORDS:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": question_type,
            "student_level": student_level,
            "status": "Unreadable",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Answer text could not be read clearly. Please upload a clearer file.",
            "student_extracted_text": student_text,
            "llm_used": False,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
        }

    if student_info.get("needs_ocr") and not student_text:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": question_type,
            "student_level": student_level,
            "status": "Unreadable",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "This PDF looks scanned. OCR is required (install pdf2image + poppler) or upload a clearer file.",
            "student_extracted_text": student_text,
            "llm_used": False,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
        }

    
    if question_type == "mixed":
        # Process each question type separately and combine results
        mcq_results = []
        narrative_results = []
        
        # Extract ALL MCQ answers from student text with question numbers
        student_answers_by_qid = extract_mcq_answers_with_qid(student_text)
        
        # Extract MCQ answers from student text for each MCQ question
        for q in parsed_questions:
            if q.get('type') == 'mcq':
                qid = q.get('qid', '')
                q_num = qid.replace('Q', '').strip() if qid else ''
                
                # Try to get answer by question number first
                chosen = student_answers_by_qid.get(qid) or student_answers_by_qid.get(f"Q{q_num}")
                
                # Fallback to old method if no question number found
                if not chosen:
                    chosen = extract_mcq_choice(student_text)
                
                correct = q.get('correct_answer') or extract_correct_mcq_from_prompt(q.get('question', ''))
                
                if correct and chosen:
                    is_correct = (chosen.lower().strip() == correct.lower().strip())
                    mcq_results.append({
                        'qid': qid,
                        'correct': is_correct,
                        'chosen': chosen,
                        'correct_answer': correct
                    })
        
        # For narrative questions, use AI to generate reference
        narrative_questions = [q for q in parsed_questions if q.get('type') == 'narrative']
        
        if narrative_questions and gemini_client:
            # Combine narrative questions into one prompt for AI
            narrative_prompt_text = "\n".join([
                f"{q.get('qid')}: {q.get('question')}" for q in narrative_questions
            ])
            
            ai_prompt = (
                f"STUDENT_LEVEL: {student_level}\n"
                f"QUESTIONS:\n{narrative_prompt_text}\n\n"
                'Return ONLY valid JSON with keys: {"ai_reference_answer": string, "key_points": [string, ...]}.'
            )
            
            response_text = generate_gemini_response(
                prompt=ai_prompt,
                system_prompt=(
                    "Generate correct reference answers for homework evaluation. "
                    "Keep it aligned with the student level. Output strict JSON only."
                ),
                max_tokens=650,
                temperature=0.3,
            )
            
            if response_text:
                try:
                    m = re.search(r'\{.*\}', response_text, flags=re.S)
                    payload = json.loads(m.group(0) if m else response_text)
                    
                    ai_reference_answer = (payload.get("ai_reference_answer") or "").strip()
                    key_points = payload.get("key_points") or []
                    
                    if isinstance(key_points, list):
                        key_points = [str(x).strip() for x in key_points if str(x).strip()]
                    
                    sim = cosine_sim(student_text, ai_reference_answer)
                    covered, missing, coverage = keypoint_coverage(
                        student_text, key_points, kp_threshold=policy["kp_thr"]
                    )
                    
                    final = policy["w_sim"] * sim + policy["w_cov"] * coverage
                    match_pct = int(round(final * 100))
                    
                    narrative_results = {
                        'similarity': sim,
                        'coverage': coverage,
                        'match_percentage': match_pct,
                        'key_points_covered': covered,
                        'key_points_missing': missing
                    }
                except Exception as e:
                    narrative_results = {'error': str(e)}
        
        # Calculate combined score with level-based partial credit for MCQ
        total_mcq = len(mcq_results)
        correct_mcq = sum(1 for r in mcq_results if r.get('correct'))
        
        # Get level-based credit per question
        mcq_credit = mcq_partial_credit(student_level)
        credit_per_q = mcq_credit["credit_per_question"]
        passing_threshold = mcq_credit["passing_threshold"]
        
        # Calculate MCQ score based on level (not just binary correct/incorrect)
        mcq_score = (correct_mcq * credit_per_q) / max(1, total_mcq)
        
        narrative_score = narrative_results.get('match_percentage', 0) if narrative_results else 0
        
        # Weight: 50% MCQ, 50% Narrative (if both exist)
        if total_mcq > 0 and narrative_results and 'error' not in narrative_results:
            final_score = int((mcq_score + narrative_score) / 2)
        elif total_mcq > 0:
            final_score = mcq_score
        elif narrative_results and 'error' not in narrative_results:
            final_score = narrative_score
        else:
            final_score = 0
        
        # Determine status
        if final_score >= policy["verified"]:
            status = "Verified"
        elif final_score >= policy["partial"]:
            status = "Partial"
        else:
            status = "Needs Review"
        
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": "mixed",
            "student_level": student_level,
            "status": status,
            "match_percentage": final_score,
            "ai_generated_remark": None,
            "rule_based_remark": f"MCQ: {correct_mcq}/{total_mcq} correct. Narrative score: {narrative_score}%. (Level: {student_level}, Credit per Q: {credit_per_q}%)",
            "llm_used": bool(narrative_results and 'error' not in narrative_results),
            "student_extracted_text": student_text,
            "mcq_results": mcq_results,
            "narrative_results": narrative_results,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
            "debug": {
                "erp_row_fields": list(erp_row.keys()) if erp_row else [],
                "erp_student_level_raw": erp_row.get("student_level") or erp_row.get("level") or erp_row.get("difficulty") or erp_row.get("difficulty_level"),
                "mcq_credit_per_q": credit_per_q,
            },
        }

    elif question_type == "mcq":
        correct = extract_correct_mcq_from_prompt(prompt)
        chosen = extract_mcq_choice(student_text)
        
        # Try to extract multiple MCQ answers (for numbered questions like "1 A", "2 B")
        student_answers_by_qid = extract_mcq_answers_with_qid(student_text)
        has_multiple_mcq = len(student_answers_by_qid) > 1

        # Smart fallback: if answer looks like narrative (not MCQ), treat as narrative instead
        # This handles cases where question type is MCQ but student answered in narrative format
        answer_looks_like_narrative = (
            len(student_text.split()) > 15 and  # More than 15 words
            not has_multiple_mcq and  # Not multiple numbered MCQ answers
            not re.search(r"\b(option|answer|ans)\s*[:\-]?\s*[a-d]\b", _norm(student_text))  # No explicit option markers
        )

        # If answer looks like narrative, redirect to narrative processing
        if answer_looks_like_narrative and gemini_client:
            question_type = "narrative"
            redirect_to_narrative = True
        else:
            redirect_to_narrative = False
            
        # Handle multiple MCQ answers - grade each one
        if has_multiple_mcq:
            # Parse prompt for multiple correct answers
            parsed_questions = parse_questions_from_prompt(prompt)
            mcq_questions_with_answers = [q for q in parsed_questions if q.get('type') == 'mcq' and q.get('correct_answer')]
            
            # If we have correct answers for multiple questions, grade them
            if mcq_questions_with_answers:
                correct_count = 0
                total_count = len(student_answers_by_qid)
                mcq_results = []
                
                for qid, student_ans in student_answers_by_qid.items():
                    # Find matching correct answer
                    matched = False
                    for pq in mcq_questions_with_answers:
                        pq_num = pq.get('qid', '').replace('Q', '').strip()
                        qid_num = qid.replace('Q', '').strip()
                        if pq_num == qid_num:
                            is_correct = student_ans.lower() == pq.get('correct_answer', '').lower()
                            if is_correct:
                                correct_count += 1
                            mcq_results.append({
                                'qid': qid,
                                'chosen': student_ans,
                                'correct_answer': pq.get('correct_answer'),
                                'correct': is_correct
                            })
                            matched = True
                            break
                    if not matched:
                        mcq_results.append({
                            'qid': qid,
                            'chosen': student_ans,
                            'correct_answer': None,
                            'correct': False
                        })
                
                # Calculate score based on level
                mcq_credit = mcq_partial_credit(student_level)
                credit_per_q = mcq_credit["credit_per_question"]
                match_percentage = int((correct_count * credit_per_q) / max(1, total_count))
                passing_threshold = mcq_credit["passing_threshold"]
                status = "Verified" if match_percentage >= passing_threshold else "Needs Review"
                
                return {
                    "student_id": student_id,
                    "homework_id": homework_id,
                    "sub_institute_id": sub_institute_id,
                    "syear": syear,
                    "question_type": "mcq",
                    "student_level": student_level,
                    "status": status,
                    "match_percentage": match_percentage,
                    "ai_generated_remark": None,
                    "rule_based_remark": f"Multiple MCQ: {correct_count}/{total_count} correct. Score: {match_percentage}% (Level: {student_level})",
                    "student_extracted_text": student_text,
                    "llm_used": False,
                    "debug": {"student_answers": student_answers_by_qid, "mcq_results": mcq_results},
                    "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
                }
            else:
                # No correct answers in prompt - return needs review with extracted answers
                return {
                    "student_id": student_id,
                    "homework_id": homework_id,
                    "sub_institute_id": sub_institute_id,
                    "syear": syear,
                    "question_type": "mcq",
                    "student_level": student_level,
                    "status": "Needs Review",
                    "match_percentage": 0,
                    "ai_generated_remark": None,
                    "rule_based_remark": f"Found {len(student_answers_by_qid)} MCQ answers but no correct answers in prompt. Include 'Correct: B' for each question.",
                    "student_extracted_text": student_text,
                    "llm_used": False,
                    "debug": {"student_answers": student_answers_by_qid, "correct_answers_in_prompt": False},
                    "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
                }

        if redirect_to_narrative:
            pass  # Will continue to narrative handling
        elif not correct:
            return {
                "student_id": student_id,
                "homework_id": homework_id,
                "sub_institute_id": sub_institute_id,
                "syear": syear,
                "question_type": "mcq",
                "student_level": student_level,
                "status": "Needs Review",
                "match_percentage": 0,
                "ai_generated_remark": None,
                "rule_based_remark": "MCQ correct option not found in prompt. Include 'Correct: B' or similar in prompt.",
                "student_extracted_text": student_text,
                "llm_used": False,
                "debug": {"correct": correct, "chosen": chosen},
                "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
            }
        elif not chosen:
            return {
                "student_id": student_id,
                "homework_id": homework_id,
                "sub_institute_id": sub_institute_id,
                "syear": syear,
                "question_type": "mcq",
                "student_level": student_level,
                "status": "Needs Review",
                "match_percentage": 0,
                "ai_generated_remark": None,
                "rule_based_remark": "Student option (A/B/C/D) not detected clearly.",
                "student_extracted_text": student_text,
                "llm_used": False,
                "debug": {"correct": correct, "chosen": chosen},
                "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
            }

        # Only process MCQ validation if not redirecting to narrative
        if not redirect_to_narrative:
            is_correct = (chosen == correct)
            
            # Get level-based credit
            mcq_credit = mcq_partial_credit(student_level)
            credit_per_q = mcq_credit["credit_per_question"]
            
            # Calculate score based on level
            match_percentage = credit_per_q if is_correct else 0
            
            # Determine status based on level threshold
            passing_threshold = mcq_credit["passing_threshold"]
            status = "Verified" if match_percentage >= passing_threshold else "Needs Review"
            
            return {
                "student_id": student_id,
                "homework_id": homework_id,
                "sub_institute_id": sub_institute_id,
                "syear": syear,
                "question_type": "mcq",
                "student_level": student_level,
                "status": status,
                "match_percentage": match_percentage,
                "ai_generated_remark": None,
                "rule_based_remark": f"{'Correct' if is_correct else 'Incorrect'}. Score: {match_percentage}% (Level: {student_level}, Credit per Q: {credit_per_q}%)",
                "student_extracted_text": student_text,
                "llm_used": False,
                "debug": {"correct": correct, "chosen": chosen, "level": student_level, "credit_per_q": credit_per_q},
                "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
            }

 
    if gemini_client is None:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": "narrative",
            "student_level": student_level,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Gemini not configured. Check /health/llm.",
            "llm_used": False,
            "llm_error": parse_gemini_error(GEMINI_LAST_ERROR),
            "student_extracted_text": student_text,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
        }

    user_prompt = (
        f"STUDENT_LEVEL: {student_level}\n"
        f"QUESTION:\n{prompt.strip()}\n\n"
        'Return ONLY valid JSON with keys: {"ai_reference_answer": string, "key_points": [string, ...]}.'
    )

    response_text = generate_gemini_response(
        prompt=user_prompt,
        system_prompt=(
            "Generate a correct reference answer for homework evaluation. "
            "Keep it aligned with the student level. Output strict JSON only."
        ),
        max_tokens=650,
        temperature=0.3,
    )

    if not response_text:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": "narrative",
            "student_level": student_level,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Gemini failed. Check /health/llm.",
            "llm_used": False,
            "llm_error": parse_gemini_error(GEMINI_LAST_ERROR),
            "student_extracted_text": student_text,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
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
            "question_type": "narrative",
            "student_level": student_level,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Gemini returned non-JSON output.",
            "llm_used": False,
            "llm_error": {"ok": False, "error_type": "GEMINI_BAD_JSON", "message": str(e), "raw": response_text[:800]},
            "student_extracted_text": student_text,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
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
            "question_type": "narrative",
            "student_level": student_level,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "AI returned empty reference answer.",
            "llm_used": True,
            "student_extracted_text": student_text,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
        }

    sim = cosine_sim(student_text, ai_reference_answer)
    covered, missing, coverage = keypoint_coverage(student_text, key_points, kp_threshold=policy["kp_thr"])

    final = policy["w_sim"] * sim + policy["w_cov"] * coverage
    match_pct = int(round(final * 100))

    if match_pct >= policy["verified"]:
        status = "Verified"
    elif match_pct >= policy["partial"]:
        status = "Partial"
    else:
        status = "Needs Review"

    # Short remark (Gemini), fallback to rule-based
    remark_prompt = (
        f"Student level: {student_level}\n"
        f"Match: {match_pct}%\n"
        f"Missing key points: {missing[:6]}\n\n"
        "Write a short, factual teacher remark (2-4 lines). No marks. No overpraise."
    )

    resp2_prompt = (
        f"REFERENCE ANSWER:\n{ai_reference_answer[:900]}\n\n"
        f"STUDENT ANSWER:\n{student_text[:900]}\n\n"
        f"{remark_prompt}"
    )

    ai_generated_remark = generate_gemini_response(
        prompt=resp2_prompt,
        system_prompt="You are a strict, helpful teacher. Be concise and factual.",
        max_tokens=140,
        temperature=0.6,
    )

    rule_based_remark = None
    remark_llm_used = bool(ai_generated_remark)
    remark_llm_error = None if ai_generated_remark else (GEMINI_LAST_ERROR or "Unknown LLM error")

    if not ai_generated_remark:
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
        "question_type": "narrative",
        "student_level": student_level,
        "status": status,
        "match_percentage": match_pct,
        "ai_generated_remark": ai_generated_remark if ai_generated_remark else None,
        "rule_based_remark": rule_based_remark,
        "llm_used": True,
        "remark_llm_used": remark_llm_used,
        "remark_llm_error": remark_llm_error,
        "student_extracted_text": student_text,
        "ai_reference_answer": ai_reference_answer,
        "key_points": key_points,
        "key_points_covered": covered,
        "key_points_missing": missing,
        "debug": {
            "similarity": sim,
            "coverage": coverage,
            "policy": policy,
            "erp_row_fields": list(erp_row.keys()) if erp_row else [],
            "erp_student_level_raw": erp_row.get("student_level") or erp_row.get("level") or erp_row.get("difficulty") or erp_row.get("difficulty_level"),
        },
        "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
    }


def extract_mcq_choice(text: str) -> str:
    """
    Extract chosen option from student text:
    supports: A, (B), Option C, Ans: D, Answer: B
    """
    t = _norm(text)

    m = re.search(r"\b(answer|ans|selected)\s*[:\-]?\s*\(?\s*([a-d])\s*\)?\b", t)
    if m:
        return m.group(2)

    m2 = re.search(r"\boption\s*([a-d])\b", t)
    if m2:
        return m2.group(1)

    m3 = re.search(r"^\(?\s*([a-d])\s*\)?$", t.strip())
    if m3:
        return m3.group(1)

    # last-resort: find first standalone A/B/C/D
    m4 = re.search(r"\b([a-d])\b", t)
    if m4:
        return m4.group(1)

    return ""


def extract_mcq_answers_with_qid(text: str) -> Dict[str, str]:
    """
    Extract MCQ answers WITH question numbers from student text.
    This handles shuffled answers where question numbers are needed to match.
    
    Supports patterns like:
    - "Q1: A, Q2: C, Q3: B"
    - "Q1. A Q2. C Q3. B"
    - "1) A 2) C 3) B"
    - "Answer 1: A Answer 2: C Answer 3: B"
    - "Q1 A Q2 C Q3 B" (space separated)
    
    Returns dict like: {"Q1": "A", "Q2": "C", "Q3": "B"}
    """
    results = {}
    t = (text or "").strip()
    
    if not t:
        return results

    # Pattern 1: Q1: A, Q2. B, Q3 - C, Question 4: D
    pattern1 = re.compile(r'(Q(?:uestion)?\s*(\d+))[:.\-\s]+([a-dA-D])', re.IGNORECASE)
    for match in pattern1.finditer(t):
        qnum = match.group(2)
        answer = match.group(3).upper()
        results[f"Q{qnum}"] = answer
    
    # Pattern 2: 1) A, 2) B, 3: C (numbered without Q prefix)
    pattern2 = re.compile(r'(?:^|\s)(\d+)\s*[\):\.]\s*([a-dA-D])(?:\s|$)', re.IGNORECASE)
    for match in pattern2.finditer(t):
        qnum = match.group(1)
        answer = match.group(2).upper()
        # Only add if not already found (Q pattern takes priority)
        if f"Q{qnum}" not in results:
            results[f"Q{qnum}"] = answer
    
    # Pattern 3: "Answer for Q1 is A", "Answer to question 2: B"
    pattern3 = re.compile(r'(?:answer|ans)\s*(?:for|to)?\s*(?:Q(?:uestion)?\s*)?(\d+)\s*(?:is|was)?\s*[:\-]?\s*([a-dA-D])', re.IGNORECASE)
    for match in pattern3.finditer(t):
        qnum = match.group(1)
        answer = match.group(2).upper()
        if f"Q{qnum}" not in results:
            results[f"Q{qnum}"] = answer
    
    # Pattern 4: Line by line format like "Q1 A" or "1 A" on same line
    pattern4 = re.compile(r'(?:^|\n)\s*(Q(?:uestion)?\s*)?(\d+)\s+([a-dA-D])\s*(?:\n|\s{2,}|$)', re.IGNORECASE)
    for match in pattern4.finditer(t):
        qnum = match.group(2)
        answer = match.group(3).upper()
        if f"Q{qnum}" not in results:
            results[f"Q{qnum}"] = answer
    
    return results


def extract_correct_mcq_from_prompt(prompt: str) -> str:
    """
    This is IMPORTANT:
    Your prompt must contain correct option somewhere like:
      - Correct: B
      - Answer: C
      - correct_option: D
      - Correct Answer(s): A. Devdatta
    or JSON: {"correct_option":"B"}
    
    Supports formats:
      - "Correct Answer: A"
      - "Correct Answer(s): A. Devdatta"
      - "Correct: B"
      - "Answer: C"
    """
    p = (prompt or "").strip()
    if not p:
        return ""

    # JSON prompt support
    if p.startswith("{") and p.endswith("}"):
        try:
            obj = json.loads(p)
            for k in ("correct_option", "correct", "answer", "ans"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return extract_mcq_choice(v)
        except Exception:
            pass

    # Text prompt support - new format: "Correct Answer(s): A. Devdatta" or "Correct Answer: B"
    t = _norm(p)
    
    # Pattern 1: "Correct Answer(s): A. ..." or "Correct Answer: B. ..."
    # This handles format like "Correct Answer(s): A. Devdatta" or "Correct Answer(s):
    #    A. Devdatta"
    m1 = re.search(r"correct\s*answer\s*\(?s\)?\s*[:\.]\s*([a-d])\.?\s*", t)
    if m1:
        return m1.group(1)
    
    # Pattern 1b: Handle multi-line format where answer is on next line like:
    # "Correct Answer(s):\n   A. Devdatta"
    m1b = re.search(r"correct\s*answer\s*\(?s\)?\s*[:\.]\s*\n\s*([a-d])\.?", t)
    if m1b:
        return m1b.group(1)
    
    # Pattern 1c: Handle format with option text after letter like "Correct Answer(s): A. Devdatta"
    m1c = re.search(r"correct\s*answer\s*\(?s\)?\s*[:\.]\s*([a-d])\.", t)
    if m1c:
        return m1c.group(1)
    
    # Pattern 2: "Correct: A" or "Answer: B" (original pattern)
    m = re.search(r"\b(correct|answer|ans)\s*[:\-]?\s*\(?\s*([a-d])\s*\)?\b", t)
    if m:
        return m.group(2)

    return ""



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


def fetch_student_level_from_erp(row: Dict[str, Any]) -> str:
    """
    ERP field name is not guaranteed; try common ones.
    """
    for k in ("student_level", "level", "difficulty", "difficulty_level"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return normalize_level(v)
    return "Medium"



def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Enhanced preprocessing for better OCR on handwritten images.
    Includes adaptive thresholding, noise removal, and contrast enhancement.
    """
    # Convert to grayscale
    img = img.convert("L")
    
    w, h = img.size
    
    # Scale up for better detail (especially for handwritten)
    if max(w, h) < 2000:
        scale = 2000 / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Apply adaptive thresholding for better handwritten recognition
    from PIL import ImageFilter
    
    # Try multiple preprocessing approaches and use the best
    img_enhanced = img
    
    # Method 1: Increase contrast significantly
    img_contrast = img.point(lambda p: 255 if p > 180 else int(p * 1.5))
    
    # Method 2: Apply sharpening twice for handwritten
    img_sharp = img.filter(ImageFilter.SHARPEN)
    img_sharp = img_sharp.filter(ImageFilter.SHARPEN)
    
    # Method 3: Apply unsharp mask for edge enhancement
    img_unsharp = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # Use the sharpened version as primary
    img = img_sharp
    
    # Apply binary threshold with lower cutoff to capture lighter handwriting
    img = img.point(lambda p: 255 if p > 160 else 0)
    
    return img


def _extract_text_google_vision(image_bytes: bytes) -> str:
    """
    Extract text using Google Cloud Vision API - much better for handwriting.
    Returns empty string if API is not available.
    """
    global vision_client
    
    if not vision_client:
        return ""
    
    try:
        # Create image object
        image = vision.Image(content=image_bytes)
        
        # Use document text detection for better handwriting
        response = vision_client.document_text_detection(image=image)
        
        if response.texts:
            return "\n".join([t.description for t in response.texts])
        return ""
    except Exception as e:
        print(f"[WARN] Google Vision OCR failed: {e}")
        return ""


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

    # First try Google Cloud Vision (better for handwriting)
    if vision_client:
        gv_text = _extract_text_google_vision(image_bytes)
        if gv_text and len(gv_text.strip()) > 10:
            return _clean_extracted_text(gv_text)
    
    # Fallback to Tesseract with improved preprocessing
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image '{filename}': {e}")

    img = _preprocess_for_ocr(img)

    # Try multiple OCR configurations for better handwritten recognition
    ocr_configs = [
        "--oem 3 --psm 6",  # Default
        "--oem 3 --psm 4",  # Treat as single column
        "--oem 1 --psm 3",  # Fully automatic
    ]
    
    best_text = ""
    best_confidence = 0
    
    for config in ocr_configs:
        try:
            text = pytesseract.image_to_string(img, lang="eng", config=config)
            if text and len(text.strip()) > len(best_text.strip()):
                best_text = text
        except Exception:
            continue
    
    if not best_text:
        # Fallback to default if all fail
        try:
            best_text = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6")
        except pytesseract.TesseractNotFoundError:
            raise HTTPException(status_code=500, detail="Tesseract OCR not found. Install it / fix path.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    text = (best_text or "").strip()
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
            # Higher DPI for better handwritten OCR
            pages = convert_from_bytes(pdf_bytes, dpi=300)
            page_texts = []
            for img in pages:
                # Use the improved preprocessing
                img = _preprocess_for_ocr(img)
                
                # Try multiple OCR configs
                for config in ["--oem 3 --psm 6", "--oem 3 --psm 4", "--oem 1 --psm 3"]:
                    try:
                        t = pytesseract.image_to_string(img, lang="eng", config=config) or ""
                        if t.strip() and len(t.strip()) > 20:
                            page_texts.append(t)
                            break
                    except:
                        continue
            
            if page_texts:
                extracted = _clean_extracted_text("\n\n".join(page_texts))
            else:
                # Final fallback with default config
                img = pages[0] if pages else None
                if img:
                    img = _preprocess_for_ocr(img)
                    extracted = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6") or ""
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
            return {"text": _clean_extracted_text(extract_text_from_image(data, filename=filename)),
                    "kind": "image", "used_ocr": True, "needs_ocr": False}
        except HTTPException as e:
            return {"text": "", "kind": "image", "used_ocr": True, "needs_ocr": False, "error": e.detail}

    if is_docx:
        try:
            return {"text": _clean_extracted_text(extract_text_from_docx(data, filename=filename)),
                    "kind": "docx", "used_ocr": False, "needs_ocr": False}
        except HTTPException as e:
            return {"text": "", "kind": "docx", "used_ocr": False, "needs_ocr": False, "error": e.detail}

    if is_pdf:
        info = extract_text_from_pdf(data, filename=filename)
        return {"text": info.get("text", ""), "kind": "pdf",
                "used_ocr": bool(info.get("used_ocr", False)),
                "needs_ocr": bool(info.get("needs_ocr", False)),
                "ocr_error": info.get("ocr_error")}

    # fallback: try as image
    try:
        return {"text": _clean_extracted_text(extract_text_from_image(data, filename=filename)),
                "kind": "unknown_as_image", "used_ocr": True, "needs_ocr": False}
    except Exception:
        return {"text": "", "kind": "unknown", "used_ocr": False, "needs_ocr": False,
                "error": f"Unsupported file type: {content_type or ext or 'unknown'}"}





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


@app.post("/homework/validate")
async def homework_validate(
    student_id: int = Form(...),
    homework_id: int = Form(...),
    sub_institute_id: int = Form(...),
    syear: str = Form(...),
    prompt: str = Form(...),
    student_file: UploadFile = File(...),
):
    # 0) Fetch ERP record -> get student_level automatically
    erp_row = fetch_student_record(homework_id, student_id)
    student_level = fetch_student_level_from_erp(erp_row)
    policy = level_policy(student_level)

    # 1) Infer question_type from prompt automatically (NO EXTRA FIELD)
    # Try to parse mixed questions first
    parsed_questions = parse_questions_from_prompt(prompt)
    has_mcq = any(q.get('type') == 'mcq' for q in parsed_questions)
    has_narrative = any(q.get('type') == 'narrative' for q in parsed_questions)
    
    # Determine overall question type for backwards compatibility
    if has_mcq and has_narrative:
        question_type = "mixed"
    elif has_mcq:
        question_type = "mcq"
    elif has_narrative:
        question_type = "narrative"
    else:
        question_type = infer_question_type_from_prompt(prompt)

    # 2) Extract student text
    student_info = await extract_text_from_upload(student_file)
    student_text = (student_info.get("text") or "").strip()

    MIN_WORDS = 3 if question_type == "mcq" else 8
    if len(student_text.split()) < MIN_WORDS:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": question_type,
            "student_level": student_level,
            "status": "Unreadable",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Answer text could not be read clearly. Please upload a clearer file.",
            "student_extracted_text": student_text,
            "llm_used": False,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
        }

    if student_info.get("needs_ocr") and not student_text:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": question_type,
            "student_level": student_level,
            "status": "Unreadable",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "This PDF looks scanned. OCR is required (install pdf2image + poppler) or upload a clearer file.",
            "student_extracted_text": student_text,
            "llm_used": False,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
        }

        if question_type == "mixed":
        # Process each question type separately and combine results
         mcq_results = []
         narrative_results = []
        
        # Extract ALL MCQ answers from student text with question numbers
        student_answers_by_qid = extract_mcq_answers_with_qid(student_text)
        
        # Extract MCQ answers from student text for each MCQ question
        for q in parsed_questions:
            if q.get('type') == 'mcq':
                qid = q.get('qid', '')
                q_num = qid.replace('Q', '').strip() if qid else ''
                
                # Try to get answer by question number first
                chosen = student_answers_by_qid.get(qid) or student_answers_by_qid.get(f"Q{q_num}")
                
                # Fallback to old method if no question number found
                if not chosen:
                    chosen = extract_mcq_choice(student_text)
                
                correct = q.get('correct_answer') or extract_correct_mcq_from_prompt(q.get('question', ''))
                
                if correct and chosen:
                    is_correct = (chosen.lower().strip() == correct.lower().strip())
                    mcq_results.append({
                        'qid': qid,
                        'correct': is_correct,
                        'chosen': chosen,
                        'correct_answer': correct
                    })
        
        # For narrative questions, use AI to generate reference
        narrative_questions = [q for q in parsed_questions if q.get('type') == 'narrative']
        
        if narrative_questions and gemini_client:
            # Combine narrative questions into one prompt for AI
            narrative_prompt_text = "\n".join([
                f"{q.get('qid')}: {q.get('question')}" for q in narrative_questions
            ])
            
            ai_prompt = (
                f"STUDENT_LEVEL: {student_level}\n"
                f"QUESTIONS:\n{narrative_prompt_text}\n\n"
                'Return ONLY valid JSON with keys: {"ai_reference_answer": string, "key_points": [string, ...]}.'
            )
            
            response_text = generate_gemini_response(
                prompt=ai_prompt,
                system_prompt=(
                    "Generate correct reference answers for homework evaluation. "
                    "Keep it aligned with the student level. Output strict JSON only."
                ),
                max_tokens=650,
                temperature=0.3,
            )
            
            if response_text:
                try:
                    m = re.search(r'\{.*\}', response_text, flags=re.S)
                    payload = json.loads(m.group(0) if m else response_text)
                    
                    ai_reference_answer = (payload.get("ai_reference_answer") or "").strip()
                    key_points = payload.get("key_points") or []
                    
                    if isinstance(key_points, list):
                        key_points = [str(x).strip() for x in key_points if str(x).strip()]
                    
                    sim = cosine_sim(student_text, ai_reference_answer)
                    covered, missing, coverage = keypoint_coverage(
                        student_text, key_points, kp_threshold=policy["kp_thr"]
                    )
                    
                    final = policy["w_sim"] * sim + policy["w_cov"] * coverage
                    match_pct = int(round(final * 100))
                    
                    narrative_results = {
                        'similarity': sim,
                        'coverage': coverage,
                        'match_percentage': match_pct,
                        'key_points_covered': covered,
                        'key_points_missing': missing
                    }
                except Exception as e:
                    narrative_results = {'error': str(e)}
        
        # Calculate combined score with level-based partial credit for MCQ
        total_mcq = len(mcq_results)
        correct_mcq = sum(1 for r in mcq_results if r.get('correct'))
        
        # Get level-based credit per question
        mcq_credit = mcq_partial_credit(student_level)
        credit_per_q = mcq_credit["credit_per_question"]
        passing_threshold = mcq_credit["passing_threshold"]
        
        # Calculate MCQ score based on level (not just binary correct/incorrect)
        mcq_score = (correct_mcq * credit_per_q) / max(1, total_mcq)
        
        narrative_score = narrative_results.get('match_percentage', 0) if narrative_results else 0
        
        # Weight: 50% MCQ, 50% Narrative (if both exist)
        if total_mcq > 0 and narrative_results and 'error' not in narrative_results:
            final_score = int((mcq_score + narrative_score) / 2)
        elif total_mcq > 0:
            final_score = mcq_score
        elif narrative_results and 'error' not in narrative_results:
            final_score = narrative_score
        else:
            final_score = 0
        
        # Determine status
        if final_score >= policy["verified"]:
            status = "Verified"
        elif final_score >= policy["partial"]:
            status = "Partial"
        else:
            status = "Needs Review"
        
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": "mixed",
            "student_level": student_level,
            "status": status,
            "match_percentage": final_score,
            "ai_generated_remark": None,
            "rule_based_remark": f"MCQ: {correct_mcq}/{total_mcq} correct. Narrative score: {narrative_score}%. (Level: {student_level}, Credit per Q: {credit_per_q}%)",
            "llm_used": bool(narrative_results and 'error' not in narrative_results),
            "student_extracted_text": student_text,
            "mcq_results": mcq_results,
            "narrative_results": narrative_results,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
            "debug": {
                "erp_row_fields": list(erp_row.keys()) if erp_row else [],
                "erp_student_level_raw": erp_row.get("student_level") or erp_row.get("level") or erp_row.get("difficulty") or erp_row.get("difficulty_level"),
                "mcq_credit_per_q": credit_per_q,
            },
        }

    elif question_type == "mcq":
        correct = extract_correct_mcq_from_prompt(prompt)
        chosen = extract_mcq_choice(student_text)
        
        # Try to extract multiple MCQ answers (for numbered questions like "1 A", "2 B")
        student_answers_by_qid = extract_mcq_answers_with_qid(student_text)
        has_multiple_mcq = len(student_answers_by_qid) > 1

        # Smart fallback: if answer looks like narrative (not MCQ), treat as narrative instead
        # This handles cases where question type is MCQ but student answered in narrative format
        answer_looks_like_narrative = (
            len(student_text.split()) > 15 and  # More than 15 words
            not has_multiple_mcq and  # Not multiple numbered MCQ answers
            not re.search(r"\b(option|answer|ans)\s*[:\-]?\s*[a-d]\b", _norm(student_text))  # No explicit option markers
        )

        # If answer looks like narrative, redirect to narrative processing
        if answer_looks_like_narrative and gemini_client:
            question_type = "narrative"
            redirect_to_narrative = True
        else:
            redirect_to_narrative = False
            
        # Handle multiple MCQ answers - grade each one
        if has_multiple_mcq:
            # Parse prompt for multiple correct answers
            parsed_questions = parse_questions_from_prompt(prompt)
            mcq_questions_with_answers = [q for q in parsed_questions if q.get('type') == 'mcq' and q.get('correct_answer')]
            
            # If we have correct answers for multiple questions, grade them
            if mcq_questions_with_answers:
                correct_count = 0
                total_count = len(student_answers_by_qid)
                mcq_results = []
                
                for qid, student_ans in student_answers_by_qid.items():
                    # Find matching correct answer
                    matched = False
                    for pq in mcq_questions_with_answers:
                        pq_num = pq.get('qid', '').replace('Q', '').strip()
                        qid_num = qid.replace('Q', '').strip()
                        if pq_num == qid_num:
                            is_correct = student_ans.lower() == pq.get('correct_answer', '').lower()
                            if is_correct:
                                correct_count += 1
                            mcq_results.append({
                                'qid': qid,
                                'chosen': student_ans,
                                'correct_answer': pq.get('correct_answer'),
                                'correct': is_correct
                            })
                            matched = True
                            break
                    if not matched:
                        mcq_results.append({
                            'qid': qid,
                            'chosen': student_ans,
                            'correct_answer': None,
                            'correct': False
                        })
                
                # Calculate score based on level
                mcq_credit = mcq_partial_credit(student_level)
                credit_per_q = mcq_credit["credit_per_question"]
                match_percentage = int((correct_count * credit_per_q) / max(1, total_count))
                passing_threshold = mcq_credit["passing_threshold"]
                status = "Verified" if match_percentage >= passing_threshold else "Needs Review"
                
                return {
                    "student_id": student_id,
                    "homework_id": homework_id,
                    "sub_institute_id": sub_institute_id,
                    "syear": syear,
                    "question_type": "mcq",
                    "student_level": student_level,
                    "status": status,
                    "match_percentage": match_percentage,
                    "ai_generated_remark": None,
                    "rule_based_remark": f"Multiple MCQ: {correct_count}/{total_count} correct. Score: {match_percentage}% (Level: {student_level})",
                    "student_extracted_text": student_text,
                    "llm_used": False,
                    "debug": {"student_answers": student_answers_by_qid, "mcq_results": mcq_results},
                    "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
                }
            else:
                # No correct answers in prompt - return needs review with extracted answers
                return {
                    "student_id": student_id,
                    "homework_id": homework_id,
                    "sub_institute_id": sub_institute_id,
                    "syear": syear,
                    "question_type": "mcq",
                    "student_level": student_level,
                    "status": "Needs Review",
                    "match_percentage": 0,
                    "ai_generated_remark": None,
                    "rule_based_remark": f"Found {len(student_answers_by_qid)} MCQ answers but no correct answers in prompt. Include 'Correct: B' for each question.",
                    "student_extracted_text": student_text,
                    "llm_used": False,
                    "debug": {"student_answers": student_answers_by_qid, "correct_answers_in_prompt": False},
                    "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
                }

        if redirect_to_narrative:
            pass  # Will continue to narrative handling
        elif not correct:
            return {
                "student_id": student_id,
                "homework_id": homework_id,
                "sub_institute_id": sub_institute_id,
                "syear": syear,
                "question_type": "mcq",
                "student_level": student_level,
                "status": "Needs Review",
                "match_percentage": 0,
                "ai_generated_remark": None,
                "rule_based_remark": "MCQ correct option not found in prompt. Include 'Correct: B' or similar in prompt.",
                "student_extracted_text": student_text,
                "llm_used": False,
                "debug": {"correct": correct, "chosen": chosen},
                "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
            }
        elif not chosen:
            return {
                "student_id": student_id,
                "homework_id": homework_id,
                "sub_institute_id": sub_institute_id,
                "syear": syear,
                "question_type": "mcq",
                "student_level": student_level,
                "status": "Needs Review",
                "match_percentage": 0,
                "ai_generated_remark": None,
                "rule_based_remark": "Student option (A/B/C/D) not detected clearly.",
                "student_extracted_text": student_text,
                "llm_used": False,
                "debug": {"correct": correct, "chosen": chosen},
                "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
            }

        # Only process MCQ validation if not redirecting to narrative
        if not redirect_to_narrative:
            is_correct = (chosen == correct)
            
            # Get level-based credit
            mcq_credit = mcq_partial_credit(student_level)
            credit_per_q = mcq_credit["credit_per_question"]
            
            # Calculate score based on level
            match_percentage = credit_per_q if is_correct else 0
            
            # Determine status based on level threshold
            passing_threshold = mcq_credit["passing_threshold"]
            status = "Verified" if match_percentage >= passing_threshold else "Needs Review"
            
            return {
                "student_id": student_id,
                "homework_id": homework_id,
                "sub_institute_id": sub_institute_id,
                "syear": syear,
                "question_type": "mcq",
                "student_level": student_level,
                "status": status,
                "match_percentage": match_percentage,
                "ai_generated_remark": None,
                "rule_based_remark": f"{'Correct' if is_correct else 'Incorrect'}. Score: {match_percentage}% (Level: {student_level}, Credit per Q: {credit_per_q}%)",
                "student_extracted_text": student_text,
                "llm_used": False,
                "debug": {"correct": correct, "chosen": chosen, "level": student_level, "credit_per_q": credit_per_q},
                "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
            }

    
    if gemini_client is None:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": "narrative",
            "student_level": student_level,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Gemini not configured. Check /health/llm.",
            "llm_used": False,
            "llm_error": parse_gemini_error(GEMINI_LAST_ERROR),
            "student_extracted_text": student_text,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
        }

    user_prompt = (
        f"STUDENT_LEVEL: {student_level}\n"
        f"QUESTION:\n{prompt.strip()}\n\n"
        'Return ONLY valid JSON with keys: {"ai_reference_answer": string, "key_points": [string, ...]}.'
    )

    response_text = generate_gemini_response(
        prompt=user_prompt,
        system_prompt=(
            "Generate a correct reference answer for homework evaluation. "
            "Keep it aligned with the student level. Output strict JSON only."
        ),
        max_tokens=650,
        temperature=0.3,
    )

    if not response_text:
        return {
            "student_id": student_id,
            "homework_id": homework_id,
            "sub_institute_id": sub_institute_id,
            "syear": syear,
            "question_type": "narrative",
            "student_level": student_level,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Gemini failed. Check /health/llm.",
            "llm_used": False,
            "llm_error": parse_gemini_error(GEMINI_LAST_ERROR),
            "student_extracted_text": student_text,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
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
            "question_type": "narrative",
            "student_level": student_level,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "Gemini returned non-JSON output.",
            "llm_used": False,
            "llm_error": {"ok": False, "error_type": "GEMINI_BAD_JSON", "message": str(e), "raw": response_text[:800]},
            "student_extracted_text": student_text,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
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
            "question_type": "narrative",
            "student_level": student_level,
            "status": "Needs Review",
            "match_percentage": 0,
            "ai_generated_remark": None,
            "rule_based_remark": "AI returned empty reference answer.",
            "llm_used": True,
            "student_extracted_text": student_text,
            "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
        }

    sim = cosine_sim(student_text, ai_reference_answer)
    covered, missing, coverage = keypoint_coverage(student_text, key_points, kp_threshold=policy["kp_thr"])

    final = policy["w_sim"] * sim + policy["w_cov"] * coverage
    match_pct = int(round(final * 100))

    if match_pct >= policy["verified"]:
        status = "Verified"
    elif match_pct >= policy["partial"]:
        status = "Partial"
    else:
        status = "Needs Review"

    # Short remark (Gemini), fallback to rule-based
    remark_prompt = (
        f"Student level: {student_level}\n"
        f"Match: {match_pct}%\n"
        f"Missing key points: {missing[:6]}\n\n"
        "Write a short, factual teacher remark (2-4 lines). No marks. No overpraise."
    )

    resp2_prompt = (
        f"REFERENCE ANSWER:\n{ai_reference_answer[:900]}\n\n"
        f"STUDENT ANSWER:\n{student_text[:900]}\n\n"
        f"{remark_prompt}"
    )

    ai_generated_remark = generate_gemini_response(
        prompt=resp2_prompt,
        system_prompt="You are a strict, helpful teacher. Be concise and factual.",
        max_tokens=140,
        temperature=0.6,
    )

    rule_based_remark = None
    remark_llm_used = bool(ai_generated_remark)
    remark_llm_error = None if ai_generated_remark else (GEMINI_LAST_ERROR or "Unknown LLM error")

    if not ai_generated_remark:
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
        "question_type": "narrative",
        "student_level": student_level,
        "status": status,
        "match_percentage": match_pct,
        "ai_generated_remark": ai_generated_remark if ai_generated_remark else None,
        "rule_based_remark": rule_based_remark,
        "llm_used": True,
        "remark_llm_used": remark_llm_used,
        "remark_llm_error": remark_llm_error,
        "student_extracted_text": student_text,
        "ai_reference_answer": ai_reference_answer,
        "key_points": key_points,
        "key_points_covered": covered,
        "key_points_missing": missing,
        "debug": {
            "similarity": sim,
            "coverage": coverage,
            "policy": policy,
            "erp_row_fields": list(erp_row.keys()) if erp_row else [],
            "erp_student_level_raw": erp_row.get("student_level") or erp_row.get("level") or erp_row.get("difficulty") or erp_row.get("difficulty_level"),
        },
        "extraction": {"student": {k: v for k, v in student_info.items() if k != "text"}},
    }

