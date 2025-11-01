import re
import logging
from typing import List, Optional, Tuple



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_whitespace(text:str)->str:

    if not text:
        return ""
    
    text = re.sub(r' +', ' ', text) #miltiple space to single
    text = re.sub(r'\n\s*\n+', '\n', text)#multiple newLines with single newLine
    lines = [line.strip() for line in text.split('\n')]# remove leading/trailong whitespace form each line
    lines = [line for line in lines if line]#remove empty lines
    return '\n'.join(lines)


def extract_amounts(text:str,currency_symbols:str= r'[$€£¥₹]')-> List[float]:

    amounts=[]

    #pattern 1:currency symbols followed by numbers
    pattern1 = rf'{currency_symbols}\s*?([\d,]+\.?\d{{0,2}})'
    matches1 = re.findall(pattern1, text)
    
    # Pattern 2: Amount followed by currency 
    pattern2 = r'([\d,]+\.?\d{0,2})\s*?(?:USD|EUR|GBP|INR|CAD)?'
    matches2 = re.findall(pattern2, text)

    all_matches = matches1 + matches2

    for match in all_matches:
        try:
            cleaned= match.replace(',','')
            amount= float(cleaned)

            #filter out unrealistic amounts
            if 0.01 <= amount <=1_000_000:
                amounts.append(amount)
        except(ValueError,AttributeError):
            continue
    return amounts  

def extract_total_amount(text: str) -> Optional[float]:
   
    total_keywords = [
        r'total\s*\(?\w*\)?',
        r'grand\s*total',
        r'amount\s*due',
        r'balance\s*due',
        r'net\s*amount',
        r'invoice\s*total'
    ]
    
    for keyword in total_keywords:
        pattern = rf'{keyword}\s*:?\s*[$€£¥₹]?\s*([\d,]+\.?\d{{0,2}})'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            try:
                amount_str = match.group(1).replace(',', '')
                return float(amount_str)
            except (ValueError, AttributeError):
                continue
    
    amounts = extract_amounts(text)
    if amounts:
        logger.warning("Total not found by keyword, returning largest amount")
        return max(amounts)
    
    return None

def guess_vendor(text:str,min_length:int=3,max_length:int =100)->Optional[str]:

    if not text:
        return None
    
    skip_keywords = [
        'invoice', 'receipt', 'bill', 'statement',
        'date', 'total', 'amount', 'subtotal', 'tax',
        'page', 'paid', 'due', 'number', 'no', '#',
        'attn', 'attention', 'to', 'from', 'bill to',
        'ship to', 'customer', 'qty', 'quantity',
        'description', 'price', 'unit'
    ]

    lines= text.split('\n')
    
    for line in lines:
        line=line.strip()

        if len(line) < min_length or len(line) > max_length:
            continue
        
        # Skip if line contains only numbers or special characters
        if re.match(r'^[\d\s\W]+$', line):
            continue
        
        # Skip if line contains skip keywords
        if any(keyword in line.lower() for keyword in skip_keywords):
            continue
        
        # Skip if line is mostly uppercase (might be a header)
        if line.isupper() and len(line) > 20:
            continue
        
        # This line looks like a vendor name
        return line[:max_length]
    
    return None

def clean_text_for_model(text: str, lowercase: bool = True) -> str:

    if not text:
        return ""
    
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    # Remove extra punctuation (keep basic punctuation)
    text = re.sub(r'[^\w\s.,!?;:\-]', '', text)
    
    # Remove multiple punctuation marks
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    return text.strip()


def extract_invoice_metadata(text:str)-> dict:

    metadata = {
        'vendor': guess_vendor(text),
        'total': extract_total_amount(text),
        'all_amounts': extract_amounts(text),
        'cleaned_text': clean_text_for_model(text, lowercase=False),
        'normalized_text': normalize_whitespace(text),
        'char_count': len(text),
        'line_count': len(text.split('\n'))
    }
    
    return metadata


#for the batch processing
def process_invoice_batch(texts: List[str]) -> List[dict]:
   
    results = []
    
    for idx, text in enumerate(texts):
        try:
            metadata = extract_invoice_metadata(text)
            metadata['index'] = idx
            results.append(metadata)
        except Exception as e:
            logger.error(f"Error processing invoice {idx}: {str(e)}")
            results.append({
                'index': idx,
                'error': str(e),
                'vendor': None,
                'total': None
            })
    
    return results




