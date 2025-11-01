import pytesseract
import cv2
import pdfplumber
import json
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime

# Config logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Tesseract path for Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import platform
import shutil

# Automatically detect tesseract path based on OS
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract_path = shutil.which("tesseract")
    if pytesseract_path:
        pytesseract.pytesseract.tesseract_cmd = pytesseract_path
    else:
        # fallback for safety
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

print("Using Tesseract at:", pytesseract.pytesseract.tesseract_cmd)



class OCRUtility:
    
    
    SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.tiff', '.webp', '.bmp'} 
    SUPPORTED_DOCUMENT_FORMATS = {'.pdf'}

    def __init__(self, tesseract_path: Optional[str] = None):
      
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Verify Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            logger.info(" OCR Utility initialized - Tesseract found")
        except pytesseract.TesseractNotFoundError:
            logger.error(" Tesseract not found! Please install it from: https://github.com/UB-Mannheim/tesseract/wiki")
            raise

    @staticmethod
    def preprocess_image(path: Path) -> cv2.typing.MatLike:  # Fixed: Added type hint
       
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read image file: {path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get black and white image
        _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return th

    @staticmethod
    def extract_text_from_image(path: Path, config: str = "--psm 6") -> str:  # Fixed: Added Path type hint
       
        try:
            path = Path(path)
            logger.info(f"Extracting text from image: {path}")
            
            img = OCRUtility.preprocess_image(path)
            text = pytesseract.image_to_string(img, config=config)
            
            char_count = len(text.strip())
            logger.info(f" Extracted {char_count} characters from image")
            
            return text.strip()
        
        except Exception as e:
            logger.error(f" Error extracting text from image {path}: {str(e)}")
            raise

    @staticmethod
    def extract_text_from_pdf(path: Path, use_ocr_fallback: bool = True) -> str:
      
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        text = ""
        
        try:
            logger.info(f"Extracting text from PDF: {path}")

            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                
                for idx, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()

                        if page_text and page_text.strip():
                            text += page_text + "\n"
                            logger.info(f"Page {idx}/{total_pages}: Direct text extraction successful")
                        
                        elif use_ocr_fallback:
                            logger.info(f"Page {idx}/{total_pages}: Using OCR fallback")
                            pil_img = page.to_image(resolution=300).original
                            ocr_text = pytesseract.image_to_string(pil_img, config="--psm 6")
                            text += ocr_text + "\n"
                        
                        else:
                            logger.warning(f"Page {idx}/{total_pages}: No text extracted")

                    except Exception as e:
                        logger.error(f" Error processing page {idx}: {str(e)}")
                        continue

            char_count = len(text.strip())
            logger.info(f" Successfully extracted {char_count} characters from PDF ({total_pages} pages)")
            
            return text.strip()

        except Exception as e:
            logger.error(f" Error reading PDF {path}: {str(e)}")
            raise

    def extract_from_file(self, path: Path, output_format: str = 'text') -> Dict:
    
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        result = {
            'file': str(path),
            'filename': path.name,
            'status': 'success',
            'text': '',
            'char_count': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if suffix in self.SUPPORTED_IMAGE_FORMATS:
                result['file_type'] = 'image'
                result['text'] = self.extract_text_from_image(path)
            
            elif suffix in self.SUPPORTED_DOCUMENT_FORMATS:
                result['file_type'] = 'pdf'
                result['text'] = self.extract_text_from_pdf(path)
            
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            result['char_count'] = len(result['text'])
        
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f" Error processing {path}: {str(e)}")
        
        return result
    
    def batch_extract(self, directory: Path, output_file: Optional[Path] = None) -> Dict:
       
        directory = Path(directory)
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        results = {
            'directory': str(directory),
            'timestamp': datetime.now().isoformat(),
            'files': [],
            'summary': {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'total_characters': 0
            }
        }
        
        # Find all supported files
        all_supported = self.SUPPORTED_IMAGE_FORMATS | self.SUPPORTED_DOCUMENT_FORMATS
        files = [f for f in directory.rglob('*') if f.suffix.lower() in all_supported]
        
        logger.info(f"Found {len(files)} files to process in {directory}")
        
        for file_path in files:
            result = self.extract_from_file(file_path)
            results['files'].append(result)
            results['summary']['total_files'] += 1
            
            if result['status'] == 'success':
                results['summary']['successful'] += 1
                results['summary']['total_characters'] += result['char_count']
            else:
                results['summary']['failed'] += 1
        
        # Save results to file if specified
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f" Results saved to {output_file}")
        
        return results

