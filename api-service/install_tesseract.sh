#!/usr/bin/env bash
set -e

echo "🧩 Installing system dependencies..."
apt-get update
apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    python3-opencv \
    poppler-utils

echo "✅ Verifying installations..."
echo "Tesseract version:"
tesseract --version

# Test if Tesseract is working
echo "🧪 Testing Tesseract installation..."
which tesseract
ls -la /usr/bin/tesseract*

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Build completed successfully!"
