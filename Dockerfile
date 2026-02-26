<<<<<<< HEAD
FROM python:3.10-slim

# Install Tesseract OCR + Poppler + dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Set HF_SPACE environment variable for Hugging Face Spaces
ENV HF_SPACE=moncey10-homework-validation-system.hf.space

# Hugging Face Spaces expects 7860 by default
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
=======
FROM python:3.10-slim

# System deps (Tesseract + basic libs for PIL)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# HF Spaces uses 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
>>>>>>> cdb5b148e5facdea1aec264a5b4d0b6293132b6e
