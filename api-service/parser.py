import re
import logging
from datetime import datetime
from typing import Dict, Optional, List
from dateutil import parser as dateparser

# Configure logging
logger = logging.getLogger(__name__)


class InvoiceParser:
   
    
    # Common currency symbols
    CURRENCY_SYMBOLS = r'[₹$€£¥]'
    
    # Common invoice field patterns
    PATTERNS = {
        'invoice_number': r'(?:Invoice|Invoice #|Invoice No|Inv #|Inv No)[:\s]+([A-Za-z0-9\-]+)',
        'po_number': r'(?:PO|PO #|PO No|Purchase Order)[:\s]+([A-Za-z0-9\-]+)',
        'gst_number': r'(?:GST|GSTIN|Tax ID)[:\s]+([0-9A-Z]{15})',
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'(?:\+\d{1,3}[-.\s]?)?\d{10,}',
    }
    
    def __init__(self):
        
        logger.info("Invoice Parser initialized")
    
    @staticmethod
    def find_date(text: str) -> Optional[str]:
       
        try:
            # Try common date formats: DD/MM/YYYY, MM-DD-YYYY, DD-MM-YYYY
            patterns = [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # DD/MM/YYYY or MM/DD/YYYY
                r'(\d{4}-\d{2}-\d{2})',  # ISO format
                r'(?:Date|Date:)[:\s]*([^\n]+)',  # "Date: ..." format
            ]
            
            for pattern in patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    date_str = m.group(1)
                    try:
                        parsed_date = dateparser.parse(date_str, dayfirst=True)
                        return parsed_date.date().isoformat()
                    except Exception as e:
                        logger.warning(f"Failed to parse date '{date_str}': {e}")
                        continue
            
            return None
        
        except Exception as e:
            logger.error(f"Error finding date: {e}")
            return None
    
    @staticmethod
    def find_due_date(text: str) -> Optional[str]:
       
        try:
            # Look for "Due Date:" pattern
            m = re.search(r'(?:Due Date|Due)[:\s]*([^\n]+)', text, re.IGNORECASE)
            if m:
                date_str = m.group(1)
                try:
                    parsed_date = dateparser.parse(date_str, dayfirst=True)
                    return parsed_date.date().isoformat()
                except Exception:
                    pass
            
            return None
        
        except Exception as e:
            logger.error(f"Error finding due date: {e}")
            return None
    
    @staticmethod
    def find_total(text: str) -> Optional[float]:
       
        try:
            # Pattern 1: Match "Grand Total" or "Total (USD)" first (most specific)
            m = re.search(
                r'(?:Grand Total|Total\s*\([^)]+\)|Amount Due)[:\s-]*\s*([₹$€£¥]?\s?[\d,]+\.?\d{0,2})',
                text,
                re.IGNORECASE
            )
            
            if m:
                amount_str = m.group(1)
                amount = re.sub(r'[^\d.]', '', amount_str)
                if amount:
                    return float(amount)
            
            # Pattern 2: Match "Total:" (general)
            m = re.search(
                r'(?:Total)[:\s-]*\s*(?!Sub|sub)([₹$€£¥]?\s?[\d,]+\.?\d{0,2})',
                text,
                re.IGNORECASE
            )
            
            if m:
                amount_str = m.group(1)
                amount = re.sub(r'[^\d.]', '', amount_str)
                if amount:
                    return float(amount)
            
            # Pattern 3: Look for last decimal number (fallback)
            amounts = re.findall(r'[\d,]+\.?\d{2}', text)
            if amounts:
                # Take the largest amount (likely the total)
                amounts_clean = [float(re.sub(r',', '', a)) for a in amounts]
                return max(amounts_clean)
            
            return None
        
        except Exception as e:
            logger.error(f"Error finding total: {e}")
            return None
    
    @staticmethod
    def find_subtotal(text: str) -> Optional[float]:
      
        try:
            m = re.search(
                r'(?:Subtotal|Sub Total)[:\s-]*\s*([₹$€£¥]?\s?[\d,]+\.?\d{0,2})',
                text,
                re.IGNORECASE
            )
            if m:
                amount_str = m.group(1)
                amount = re.sub(r'[^\d.]', '', amount_str)
                return float(amount) if amount else None
            
            return None
        
        except Exception as e:
            logger.error(f"Error finding subtotal: {e}")
            return None
    
    @staticmethod
    def find_tax(text: str) -> Optional[float]:
       
        try:
            m = re.search(
                r'(?:Tax|GST|VAT|IGST|SGST|CGST)[:\s-]*\s*([₹$€£¥]?\s?[\d,]+\.?\d{0,2})',
                text,
                re.IGNORECASE
            )
            if m:
                amount_str = m.group(1)
                amount = re.sub(r'[^\d.]', '', amount_str)
                return float(amount) if amount else None
            
            return None
        
        except Exception as e:
            logger.error(f"Error finding tax: {e}")
            return None
    
    @staticmethod
    def find_vendor(text: str) -> Optional[str]:
      
        try:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            
            if not lines:
                return None
            
            # Filter out common irrelevant lines
            skip_keywords = ['invoice', 'date', 'total', 'amount', 'tax', 'page']
            
            for line in lines:
                if not any(keyword in line.lower() for keyword in skip_keywords):
                    if len(line) > 3:  # Avoid very short lines
                        return line[:100]  # Limit to 100 characters
            
            # Fallback to first line if all filtered
            return lines[0][:100]
        
        except Exception as e:
            logger.error(f"Error finding vendor: {e}")
            return None
    
    @staticmethod
    def find_invoice_number(text: str) -> Optional[str]:
       
        try:
            m = re.search(
                r'(?:Invoice|Invoice #|Invoice No|Inv #|Inv No)[:\s]+([A-Za-z0-9\-/]+)',
                text,
                re.IGNORECASE
            )
            return m.group(1).strip() if m else None
        
        except Exception as e:
            logger.error(f"Error finding invoice number: {e}")
            return None
    
    @staticmethod
    def find_po_number(text: str) -> Optional[str]:
        
        try:
            m = re.search(
                r'(?:PO|PO #|PO No|Purchase Order)[:\s]+([A-Za-z0-9\-/]+)',
                text,
                re.IGNORECASE
            )
            return m.group(1).strip() if m else None
        
        except Exception as e:
            logger.error(f"Error finding PO number: {e}")
            return None
    
    @staticmethod
    def find_gst_number(text: str) -> Optional[str]:
       
        try:
            m = re.search(
                r'(?:GST|GSTIN|Tax ID|Tax Number)[:\s]+([0-9A-Z]{15}|[0-9A-Z\-]{10,})',
                text,
                re.IGNORECASE
            )
            return m.group(1).strip() if m else None
        
        except Exception as e:
            logger.error(f"Error finding GST number: {e}")
            return None
    
    @staticmethod
    def find_email(text: str) -> Optional[str]:
      
        try:
            m = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
            return m.group(0) if m else None
        
        except Exception as e:
            logger.error(f"Error finding email: {e}")
            return None
    
    @staticmethod
    def find_phone(text: str) -> Optional[str]:
       
        try:
            m = re.search(r'(?:\+\d{1,3}[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{4,}', text)
            return m.group(0) if m else None
        
        except Exception as e:
            logger.error(f"Error finding phone: {e}")
            return None
    
    def parse_invoice(self, text: str) -> Dict:
      
        try:
            logger.info("Parsing invoice text")
            
            invoice_data = {
                'vendor': self.find_vendor(text),
                'invoice_number': self.find_invoice_number(text),
                'po_number': self.find_po_number(text),
                'date': self.find_date(text),
                'due_date': self.find_due_date(text),
                'subtotal': self.find_subtotal(text),
                'tax': self.find_tax(text),
                'total': self.find_total(text),
                'gst_number': self.find_gst_number(text),
                'email': self.find_email(text),
                'phone': self.find_phone(text),
                'parsed_at': datetime.now().isoformat()
            }
            
            logger.info("Invoice parsing completed successfully")
            return invoice_data
        
        except Exception as e:
            logger.error(f"Error parsing invoice: {e}")
            raise


