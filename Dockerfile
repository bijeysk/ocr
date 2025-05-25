# Use Python 3.9 slim image as base
FROM python:3.9-slim

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

# Verify Tesseract installation
RUN echo "Verifying Tesseract installation..." && \
    tesseract --version && \
    echo "Tesseract location: $(which tesseract)" && \
    echo "Contents of TESSDATA_PREFIX directory:" && \
    ls -la $TESSDATA_PREFIX && \
    echo "Testing Tesseract OCR..." && \
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

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"] 