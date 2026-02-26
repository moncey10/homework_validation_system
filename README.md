<<<<<<< HEAD
---
title: Homework Validation System
sdk: docker
app_port: 7860
---


---
title: Homework Validation System
sdk: docker
app_port: 7860
---
# Homework Validation System (FastAPI)

A backend API that validates student homework by extracting text from teacher and student files, comparing answers, and generating remarks using rule-based logic and optional AI.

---

## Features

- Upload teacher and student homework files
- OCR support for images and scanned PDFs
- Text extraction from PDF and DOCX
- Similarity matching using TF-IDF + cosine similarity
- Optional AI-generated remarks (OpenAI / Gemini)
- FastAPI Swagger documentation

---

## Tech Stack

- FastAPI
- Python
- pytesseract
- Pillow
- pypdf / pdf2image
- python-docx
- scikit-learn
- OpenAI / Gemini (optional)

---

## Project Structure

---
homework_validation_system/
│
├── app.py
├── requirements.txt
├── artifacts/
├── uploads/
├── src/
│ ├── extractors.py
│ ├── similarity.py
│ ├── llm_client.py
│ └── utils.py
└── README.md
## Installation

### 1. Create Virtual Environment
python -m venv myenv

### 2. Install Requirements
pip install -r requirements.txt
## OCR Setup (Required)

### Install Tesseract OCR

This project uses **Tesseract OCR** for extracting text from images and scanned PDFs.

#### Windows
1. Download and install Tesseract OCR.
2. Default installation path: 
3. Add this path in your code:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

### Run API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

### Swagger UI:

http://localhost:8000/docs

### Example API Response
{
  "student_id": 1,
  "homework_id": 10,
  "status": "Needs Review",
  "match_percentage": 72,
  "teacher_extracted_text": "...",
  "student_extracted_text": "...",
  "ai_generated_remark": "Good attempt but missing key points.",
  "llm_used": true
}
>>>>>>> cdb5b148e5facdea1aec264a5b4d0b6293132b6e
