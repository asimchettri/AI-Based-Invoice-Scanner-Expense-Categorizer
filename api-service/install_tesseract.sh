#!/usr/bin/env bash
set -e

echo "🧩 Installing Tesseract OCR..."
apt-get update && apt-get install -y tesseract-ocr

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt



