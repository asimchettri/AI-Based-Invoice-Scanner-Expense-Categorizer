
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional
import logging

# Import existing utilities
from ocr import OCRUtility
from parser import InvoiceParser
from classifier import InvoiceClassifier

# Import ML model with fallback
try:
    from ml_model import ExpenseClassifier, extract_amount_from_text
    ML_MODEL_AVAILABLE = True
except ImportError:
    print("Warning: ml_model.py not found. Using rule-based classification only.")
    ML_MODEL_AVAILABLE = False

#added later due to error in render while importing cleaner.py
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(BASE_DIR, "utils")

if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

print("Added utils path:", UTILS_PATH)


# Import cleaner utilities
try:
    from utils.cleaner import guess_vendor, extract_total_amount, clean_text_for_model
except ImportError:
    try:
        from cleaner import guess_vendor, extract_total_amount, clean_text_for_model
    except ImportError:
        print("Warning: cleaner.py not found. Limited functionality.")
        def guess_vendor(text): return None
        def extract_total_amount(text): return None
        def clean_text_for_model(text, **kwargs): return text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Invoice Processing API",
    description="Complete API for OCR extraction, parsing, and ML-based classification",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)



class AnalyzeResponse(BaseModel):
    
    
    # Basic Info
    vendor: Optional[str] = Field(None, description="Extracted vendor name")
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    po_number: Optional[str] = Field(None, description="Purchase order number")
    description: Optional[str] = Field(None, description="Invoice description")
    
    # Dates - BOTH FIELDS
    invoice_date: Optional[str] = Field(None, description="Invoice date")
    due_date: Optional[str] = Field(None, description="Due date")
    
    # Financial Data - ALL AMOUNTS
    amount: Optional[float] = Field(None, description="Total amount (primary)")
    total: Optional[float] = Field(None, description="Total amount (alias)")
    subtotal: Optional[float] = Field(None, description="Subtotal amount")
    tax: Optional[float] = Field(None, description="Tax amount")
    
    # Classification with Source Tracking
    category: str = Field(..., description="Predicted expense category")
    source: str = Field(..., description="Prediction source: 'model' or 'rule'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    
    # Additional Info - ALL FIELDS
    gst_number: Optional[str] = Field(None, description="GST/Tax ID")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    
    # OCR Data
    raw_text: Optional[str] = Field(None, description="Extracted OCR text")
    char_count: Optional[int] = Field(None, description="Character count")
    
    class Config:
        schema_extra = {
            "example": {
                "vendor": "ABC Corporation",
                "invoice_number": "INV-2024-001",
                "po_number": "PO-2024-001",
                "description": "Office supplies purchase",
                "invoice_date": "2024-01-15",
                "due_date": "2024-02-15",
                "amount": 2499.00,
                "total": 2499.00,
                "subtotal": 2380.00,
                "tax": 119.00,
                "category": "retail",
                "source": "model",
                "confidence": 0.92,
                "gst_number": "18AABCT1234H1Z5",
                "email": "contact@abc.com",
                "phone": "+1-555-0123",
                "raw_text": "ABC Corporation Invoice...",
                "char_count": 528
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: dict
    ml_model_loaded: bool = False




try:
    ocr_utility = OCRUtility()
    logger.info("✓ OCR Utility initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize OCR Utility: {str(e)}")
    ocr_utility = None

try:
    invoice_parser = InvoiceParser()
    logger.info("✓ Invoice Parser initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize Invoice Parser: {str(e)}")
    invoice_parser = None

try:
    invoice_classifier = InvoiceClassifier()
    logger.info("✓ Rule-based Classifier initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize Classifier: {str(e)}")
    invoice_classifier = None

# Initialize ML model with fallback
ml_classifier = None
if ML_MODEL_AVAILABLE:
    try:
        ml_classifier = ExpenseClassifier()
        logger.info(" ML Classifier with fallback initialized")
    except Exception as e:
        logger.error(f" Failed to initialize ML Classifier: {str(e)}")




@app.get("/")
async def root():
    
    return {
        "message": "Invoice Processing API - AI-Based Invoice Scanner & Expense Categorizer",
        "status": "running",
        "version": "3.0.0",
        "features": [
            "OCR text extraction from images and PDFs",
            "Complete invoice data parsing (ALL fields)",
            "ML-based expense classification with rule-based fallback",
            "Source tracking (model vs rule-based predictions)",
            "Comprehensive financial data extraction (subtotal, tax, total)"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "analyze": "/analyze (MAIN ENDPOINT - Returns ALL invoice data)",
            "process_upload": "/process-upload",
            "process_batch": "/process-batch",
            "categories": "/categories"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    
    ml_loaded = False
    if ml_classifier:
        try:
            info = ml_classifier.get_model_info()
            ml_loaded = info.get('model_loaded', False)
        except:
            pass
    
    status = {
        "status": "healthy",
        "version": "3.0.0",
        "services": {
            "ocr": ocr_utility is not None,
            "parser": invoice_parser is not None,
            "rule_classifier": invoice_classifier is not None,
            "ml_classifier": ml_classifier is not None
        },
        "ml_model_loaded": ml_loaded
    }
    
    if not ocr_utility:
        raise HTTPException(status_code=503, detail="OCR service unavailable")
    
    return status


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_invoice(file: UploadFile = File(...)):
    
    if not ocr_utility:
        raise HTTPException(status_code=503, detail="OCR service not available")
    
    # Validate file
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    file_suffix = Path(file.filename).suffix.lower()
    allowed_formats = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.webp', '.bmp'}
    
    if file_suffix not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_suffix}. Allowed: {', '.join(allowed_formats)}"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    file_path = None
    
    try:
        # Save file temporarily
        file_path = UPLOAD_DIR / file.filename
        file_path.write_bytes(contents)
        
        logger.info(f"[ANALYZE] Processing: {file.filename} ({len(contents)} bytes)")
        
      
        
        try:
            ocr_result = ocr_utility.extract_from_file(file_path)
            raw_text = ocr_result.get('text', '')
            
            if not raw_text or len(raw_text.strip()) < 10:
                raise HTTPException(status_code=422, detail="OCR failed: No text extracted")
            
            logger.info(f"[OCR] Extracted {len(raw_text)} characters")
        
        except Exception as e:
            logger.error(f"[OCR] Error: {str(e)}")
            raise HTTPException(status_code=422, detail=f"OCR failed: {str(e)}")
        
        
        parsed_data = {}
        if invoice_parser:
            try:
                parsed_data = invoice_parser.parse_invoice(raw_text)
                logger.info(f"[PARSE] Extracted vendor: {parsed_data.get('vendor')}, "
                          f"total: {parsed_data.get('total')}, "
                          f"invoice_number: {parsed_data.get('invoice_number')}")
            except Exception as e:
                logger.warning(f"[PARSE] Error: {str(e)}")
        
       
        vendor = parsed_data.get('vendor')
        if not vendor:
            vendor = guess_vendor(raw_text)
            logger.info(f"[FALLBACK] Vendor from cleaner: {vendor}")
     
        amount = parsed_data.get('total')
        if not amount:
            amount = extract_total_amount(raw_text)
            logger.info(f"[FALLBACK] Amount from cleaner: {amount}")
        
       
        category = "other"
        source = "rule"
        confidence = 0.5
        
        if ml_classifier:
            try:
                # Use ML classifier (which has built-in rule-based fallback)
                category, source, confidence = ml_classifier.predict_category(
                    vendor=vendor or "",
                    description=parsed_data.get('description', ''),
                    amount=amount or 0.0
                )
                logger.info(f"[CLASSIFY] ML prediction: {category} "
                          f"(source: {source}, confidence: {confidence:.2f})")
            
            except Exception as e:
                logger.warning(f"[CLASSIFY] ML failed: {str(e)}, using rule-based fallback")
                # Fallback to rule-based classifier
                if invoice_classifier and vendor:
                    result = invoice_classifier.categorize_invoice(vendor, raw_text)
                    category = result.get('category', 'other')
                    confidence = result.get('confidence', 0.5)
                    source = "rule"
        
        elif invoice_classifier and vendor:
            # Only rule-based classifier available
            result = invoice_classifier.categorize_invoice(vendor, raw_text)
            category = result.get('category', 'other')
            confidence = result.get('confidence', 0.5)
            source = "rule"
            logger.info(f"[CLASSIFY] Rule-based: {category} (confidence: {confidence:.2f})")
        

        response = AnalyzeResponse(
            # Basic Info
            vendor=vendor,
            invoice_number=parsed_data.get('invoice_number'),
            po_number=parsed_data.get('po_number'),
            description=parsed_data.get('description', ''),
            
            # Dates - BOTH FIELDS
            invoice_date=parsed_data.get('date'),
            due_date=parsed_data.get('due_date'),
            
            # Financial Data - ALL AMOUNTS
            amount=amount,  # Primary amount field
            total=parsed_data.get('total') or amount,  # Explicit total
            subtotal=parsed_data.get('subtotal'),
            tax=parsed_data.get('tax'),
            
            # Classification with Source Tracking
            category=category,
            source=source,
            confidence=confidence,
            
            # Additional Info - ALL FIELDS
            gst_number=parsed_data.get('gst_number'),
            email=parsed_data.get('email'),
            phone=parsed_data.get('phone'),
            
            # OCR Data
            raw_text=raw_text[:500] if raw_text else None,  # First 500 chars
            char_count=len(raw_text) if raw_text else 0
        )
        
        logger.info(f"[SUCCESS] Analysis complete: {category} ({source}) - "
                   f"vendor={vendor}, amount={amount}, invoice_number={parsed_data.get('invoice_number')}")
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except:
                pass


@app.get("/categories")
async def get_categories():
   
    if not invoice_classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    categories = invoice_classifier.get_categories()
    stats = invoice_classifier.get_category_stats()
    
    # Add ML model info if available
    ml_info = {}
    if ml_classifier:
        try:
            ml_info = ml_classifier.get_model_info()
        except:
            pass
    
    return {
        "categories": categories,
        "stats": stats,
        "ml_model_info": ml_info
    }


@app.post("/process-upload")
async def process_upload(file: UploadFile = File(...), parse: bool = True, classify: bool = True):
  
    if not ocr_utility:
        raise HTTPException(status_code=503, detail="OCR Utility not initialized")
    
    # Validate file format
    file_suffix = Path(file.filename).suffix.lower()
    allowed_formats = OCRUtility.SUPPORTED_IMAGE_FORMATS | OCRUtility.SUPPORTED_DOCUMENT_FORMATS
    
    if file_suffix not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_suffix}"
        )
    
    file_path = None
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        contents = await file.read()
        file_path.write_bytes(contents)
        
        logger.info(f"File uploaded: {file.filename} ({len(contents)} bytes)")
        
        # Extract text using OCR
        ocr_result = ocr_utility.extract_from_file(file_path)
        
        response = {
            "status": "success",
            "filename": file.filename,
            "file_type": ocr_result['file_type'],
            "char_count": ocr_result['char_count'],
            "text": ocr_result['text'],
            "timestamp": ocr_result['timestamp']
        }
        
        # Parse invoice data
        if parse and invoice_parser:
            try:
                parsed_data = invoice_parser.parse_invoice(ocr_result['text'])
                response["parsed_data"] = parsed_data
                logger.info("Invoice parsed successfully")
            except Exception as e:
                logger.error(f"Error parsing invoice: {str(e)}")
                response["parse_error"] = str(e)
        
        # Classify using ML model (with fallback)
        if classify:
            try:
                vendor = response.get("parsed_data", {}).get("vendor", "")
                
                if ml_classifier and vendor:
                    # Use ML classifier
                    category, source, confidence = ml_classifier.predict_category(
                        vendor=vendor,
                        description="",
                        amount=response.get("parsed_data", {}).get("total", 0.0)
                    )
                    
                    response["classification"] = {
                        "category": category,
                        "source": source,
                        "confidence": confidence,
                        "description": f"{category} (via {source})"
                    }
                    logger.info(f"ML Classification: {category} ({source})")
                
                elif invoice_classifier and vendor:
                    # Fallback to rule-based
                    classification = invoice_classifier.categorize_invoice(
                        vendor_name=vendor,
                        invoice_text=ocr_result['text']
                    )
                    classification['source'] = 'rule'
                    response["classification"] = classification
                    logger.info(f"Rule Classification: {classification['category']}")
                
                else:
                    response["classification_info"] = "No vendor data available"
            
            except Exception as e:
                logger.error(f"Error classifying: {str(e)}")
                response["classification_error"] = str(e)
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if file_path and file_path.exists():
            file_path.unlink()


@app.post("/process-batch")
async def process_batch(files: List[UploadFile] = File(...), parse: bool = True, classify: bool = True):
  
    if not ocr_utility:
        raise HTTPException(status_code=503, detail="OCR Utility not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed")
    
    results = {
        "status": "success",
        "batch_size": len(files),
        "files": [],
        "summary": {
            "total": len(files),
            "successful": 0,
            "failed": 0,
            "total_characters": 0,
            "categories": {},
            "sources": {"model": 0, "rule": 0}
        }
    }
    
    allowed_formats = OCRUtility.SUPPORTED_IMAGE_FORMATS | OCRUtility.SUPPORTED_DOCUMENT_FORMATS
    
    for file in files:
        file_suffix = Path(file.filename).suffix.lower()
        file_path = UPLOAD_DIR / file.filename
        
        try:
            if file_suffix not in allowed_formats:
                results['files'].append({
                    "filename": file.filename,
                    "status": "error",
                    "error": f"Unsupported format: {file_suffix}"
                })
                results['summary']['failed'] += 1
                continue
            
            contents = await file.read()
            file_path.write_bytes(contents)
            
            ocr_result = ocr_utility.extract_from_file(file_path)
            
            file_result = {
                "filename": file.filename,
                "status": ocr_result['status'],
                "file_type": ocr_result.get('file_type'),
                "char_count": ocr_result['char_count'],
                "text": ocr_result['text'] if ocr_result['status'] == 'success' else None,
            }
            
            if parse and invoice_parser and ocr_result['status'] == 'success':
                try:
                    parsed_data = invoice_parser.parse_invoice(ocr_result['text'])
                    file_result["parsed_data"] = parsed_data
                except Exception as e:
                    file_result["parse_error"] = str(e)
            
            if classify and "parsed_data" in file_result:
                try:
                    vendor = file_result["parsed_data"].get("vendor", "")
                    
                    if ml_classifier and vendor:
                        category, source, confidence = ml_classifier.predict_category(
                            vendor=vendor,
                            description="",
                            amount=file_result["parsed_data"].get("total", 0.0)
                        )
                        file_result["classification"] = {
                            "category": category,
                            "source": source,
                            "confidence": confidence
                        }
                    elif invoice_classifier and vendor:
                        classification = invoice_classifier.categorize_invoice(vendor, ocr_result['text'])
                        classification['source'] = 'rule'
                        file_result["classification"] = classification
                    
                    if "classification" in file_result:
                        category = file_result["classification"]['category']
                        source = file_result["classification"]['source']
                        results['summary']['categories'][category] = results['summary']['categories'].get(category, 0) + 1
                        results['summary']['sources'][source] = results['summary']['sources'].get(source, 0) + 1
                
                except Exception as e:
                    file_result["classification_error"] = str(e)
            
            results['files'].append(file_result)
            
            if ocr_result['status'] == 'success':
                results['summary']['successful'] += 1
                results['summary']['total_characters'] += ocr_result['char_count']
            else:
                results['summary']['failed'] += 1
            
            file_path.unlink()
        
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results['files'].append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
            results['summary']['failed'] += 1
            
            if file_path.exists():
                file_path.unlink()
    
    return results
    


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "Internal server error",
            "error_type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
