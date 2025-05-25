# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Verify Tesseract and language data installation
RUN tesseract --version && \
    echo "Checking Tesseract locations:" && \
    which tesseract && \
    echo "Checking tessdata contents:" && \
    ls -la /usr/share/tesseract-ocr/tessdata && \
    echo "Verifying eng.traineddata:" && \
    ls -la /usr/share/tesseract-ocr/tessdata/eng.traineddata && \
    echo "Testing Tesseract OCR:" && \
    echo "Hello World" > test.txt && \
    tesseract test.txt stdout

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Set environment variables for Tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata
ENV TESSERACT_PATH=/usr/bin/tesseract
ENV PYTHONUNBUFFERED=1

# Additional verification of Tesseract setup
RUN echo "Verifying final Tesseract configuration:" && \
    echo "TESSDATA_PREFIX=$TESSDATA_PREFIX" && \
    echo "TESSERACT_PATH=$TESSERACT_PATH" && \
    ls -la $TESSDATA_PREFIX/eng.traineddata

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"] 