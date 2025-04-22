FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models and initialize them
RUN python -c "import easyocr; reader = easyocr.Reader(['en'])"

# Pre-download Ultralytics models
RUN python -c "import torch; from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy application code, OCR module, and models
COPY main.py .
COPY nutrition_label_detector.pt .
COPY ocr/ ./ocr/

# Set environment variables
ENV PORT=8080

# Command to run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app