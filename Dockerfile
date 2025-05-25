# Use Python 3.9 slim image as base
FROM python:3.9.18-slim

# Set environment variables for Tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata
ENV TESSERACT_PATH=/usr/bin/tesseract
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure tessdata directory exists with correct permissions
RUN mkdir -p /usr/share/tesseract-ocr/tessdata && \
    chmod 755 /usr/share/tesseract-ocr/tessdata

# Verify Tesseract installation step by step
RUN echo "Checking Tesseract version..." && \
    tesseract --version || echo "Failed to get version"

RUN echo "Checking Tesseract location..." && \
    which tesseract || echo "Failed to find tesseract"

RUN echo "Checking TESSDATA_PREFIX contents..." && \
    ls -la /usr/share/tesseract-ocr/ || echo "Failed to list tesseract-ocr directory"

RUN echo "Checking actual tessdata directory..." && \
    ls -la /usr/share/tesseract-ocr/tessdata/ || echo "Failed to list tessdata directory"

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

# Final verification of Tesseract
RUN echo "Testing basic OCR functionality..." && \
    echo "Hello World" > test.txt && \
    tesseract test.txt stdout || echo "OCR test failed"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"] 