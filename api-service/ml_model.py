#ml_model = model loader + rule base fall back

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import joblib
import warnings


#import existing classifier
try:
    from classifier import InvoiceClassifier
    RULE_CLASSIFIER_AVAILABLE = True 
except ImportError:
    print("Warning: classifier.py not found. Rule-based classification will be limited.")
    RULE_CLASSIFIER_AVAILABLE = False
    InvoiceClassifier = None


try:
    from utils.cleaner import clean_text_for_model,extract_amounts,extract_total_amount
except ImportError:
    try:
        from utils.cleaner import clean_text_for_model,extract_amounts,extract_total_amount
    except ImportError:
        print("cleaner.py not found . using basic text cleaning")
        def clean_text_for_model(text, lowercase=True):
            return text.lower().strip() if lowercase else text.strip()
        def extract_amounts(text):
            import re
            amounts = re.findall(r'\d+\.?\d{0,2}', text)
            return [float(a) for a in amounts if a]
        def extract_total_amount(text):
            amounts = extract_amounts(text)
            return max(amounts) if amounts else None 
         
#config logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
MODEL_PATH = Path("models/classifier.joblib")
METADATA_PATH = Path("models/metadata.joblib")

CATEGORIES = [
    'travel', 'meals', 'saas', 'office', 'utilities',
    'healthcare', 'retail', 'education', 'entertainment',
    'maintenance', 'other'
]

class ExpenseClassifier:


    def __init__(self,model_path:str="models/classifier.joblib"):
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = None
        self.rule_classifier = None
        self.model_loaded = False

        self._load_model()

        #rulebase classifier as fallback
        if RULE_CLASSIFIER_AVAILABLE:
            self.rule_classifier = InvoiceClassifier()
            logger.info("Rule-based classifier initialized as fallback")
        
        logger.info(f"ExpenseClassifier initialized (ML Model: {self.model_loaded})")

    def _load_model(self)->bool:

        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found at: {self.model_path}")
                logger.info("Will use rule-based classification as fallback")
                return False
        
            # Load the model object
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ ML model loaded from: {self.model_path}")
            logger.info(f"  Model type: {type(self.model).__name__}")
            
            # Load metadata from separate file
            metadata_path = self.model_path.parent / "metadata.joblib"
            if metadata_path.exists():
                self.metadata = joblib.load(metadata_path)
                logger.info(f"  Model name: {self.metadata.get('model_name', 'Unknown')}")
                logger.info(f"  Accuracy: {self.metadata.get('accuracy', 0)*100:.2f}%")
                logger.info(f"  Trained at: {self.metadata.get('trained_at', 'Unknown')}")
                logger.info(f"  Training samples: {self.metadata.get('training_samples', 'Unknown')}")
            else:
                logger.warning("Metadata file not found, using default info")
                self.metadata = {
                    'model_name': type(self.model).__name__,
                    'accuracy': 0.9833,  # From your training results
                    'categories': ['office', 'meals', 'transport', 'groceries', 'other', 'health', 'subscription', 'maintenance', 'entertainment', 'accommodation', 'utilities', 'travel']
                }
            
            self.model_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Will use rule-based classification as fallback")
            self.model = None
            self.model_loaded = False
            return False
        
    def rule_based_classify(self, text: str) -> Tuple[str, float]:
       
        if self.rule_classifier:
            # Use InvoiceClassifier from classifier.py
            result = self.rule_classifier.categorize_invoice(
                vendor_name=text,
                invoice_text=text
            )
            return result['category'], result['confidence']
        
        
        text_lower = text.lower()
        
       
        simple_rules = {
            'travel': ['uber', 'lyft', 'taxi', 'airline', 'flight', 'hotel', 'airbnb'],
            'meals': ['restaurant', 'pizza', 'burger', 'starbucks', 'coffee', 'food'],
            'saas': ['aws', 'github', 'slack', 'zoom', 'subscription', 'software'],
            'office': ['staples', 'office', 'supplies', 'printer', 'paper'],
            'utilities': ['electric', 'power', 'internet', 'phone', 'verizon', 'at&t'],
            'healthcare': ['pharmacy', 'cvs', 'walgreens', 'medical', 'hospital'],
            'retail': ['amazon', 'walmart', 'target', 'store', 'shopping'],
            'education': ['course', 'training', 'udemy', 'coursera', 'school'],
            'entertainment': ['netflix', 'spotify', 'hulu', 'movie', 'music'],
            'maintenance': ['repair', 'maintenance', 'plumbing', 'cleaning']
        }
        
        for category, keywords in simple_rules.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category, 0.7  # Medium confidence for keyword match
        
        return 'other', 0.5  # Low confidence for default  

    def predict_category(self,vendor:str,description:str="",amount:float=0.0)->Tuple[str,str,float]:

        vendor_clean = clean_text_for_model(str(vendor)) if vendor else ""
        description_clean = clean_text_for_model(str(description)) if description else ""
        amount_str = f"amount_{int(amount)}" if amount > 0 else "amount_0"

        #combine features
        combined_text = f"{vendor_clean} {description_clean} {amount_str}"

        #try ml model first 
        if self.model_loaded and self.model:
            try:
                prediction = self.model.predict([combined_text])[0]

                #get confidence if model supports predirct_proba 
                try:
                    proba = self.model.predict_proba([combined_text])[0]
                    confidence = max(proba)
                except AttributeError:
                    confidence = 0.85  # Default confidence for models without proba
                
                logger.info(f"ML Model prediction: {prediction} (confidence: {confidence:.2f})")
                return prediction, "model", confidence
            
            except Exception as e:
                logger.error(f"Error during ML prediction: {str(e)}")
                logger.info("Falling back to rule-based classification")

        #fallback to rule-based
        category,confidence = self.rule_based_classify(combined_text)
        logger.info(f"Rule-based prediction: {category} (confidence: {confidence:.2f})")
        return category, "rule", confidence
    
    def predict_from_text(self,text :str)-> Tuple[str,str,float]:

        if self.model_loaded and self.model:
            try:
                cleaned = clean_text_for_model(text)
                prediction = self.model.predict([cleaned])[0]
                
                try:
                    proba = self.model.predict_proba([cleaned])[0]
                    confidence = max(proba)
                except AttributeError:
                    confidence = 0.85
                
                return prediction, "model", confidence
            
            except Exception as e:
                logger.error(f"Error during ML prediction: {str(e)}")
        
        # Fallback to rules
        category, confidence = self.rule_based_classify(text)
        return category, "rule", confidence
    
    
    def get_model_info(self) -> Dict:
       
        info = {
            'model_loaded': self.model_loaded,
            'model_path': str(self.model_path),
            'rule_classifier_available': self.rule_classifier is not None,
            'categories': CATEGORIES
        }
        
        if self.metadata:
            info.update({
                'model_name': self.metadata.get('model_name'),
                'accuracy': self.metadata.get('accuracy'),
                'trained_at': self.metadata.get('trained_at'),
                'training_samples': self.metadata.get('training_samples'),
                'data_source': self.metadata.get('data_source')
            })
        
        return info
    
    def reload_model(self) -> bool:
       
        logger.info("Reloading model...")
        return self._load_model()


# Helper functions for amount extraction
def extract_amount_from_text(text: str) -> Optional[float]:
   
    return extract_total_amount(text)


def extract_all_amounts(text: str) -> list:
   
    return extract_amounts(text)


def classify_expense(vendor: str, description: str = "", amount: float = 0.0) -> Dict:
   
    classifier = ExpenseClassifier()
    category, source, confidence = classifier.predict_category(vendor, description, amount)
    
    return {
        'category': category,
        'source': source,
        'confidence': confidence,
        'model_loaded': classifier.model_loaded
    }

#can comment out later when backend starts to use it

# Test function
def test_classifier():
    """
    Test the expense classifier with sample data.
    """
    print("=" * 70)
    print("Testing ExpenseClassifier")
    print("=" * 70)
    
    # Initialize classifier
    classifier = ExpenseClassifier()
    
    # Show model info
    print("\nModel Information:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test cases
    test_cases = [
        {"vendor": "Uber", "description": "ride downtown", "amount": 18.50},
        {"vendor": "Starbucks", "description": "coffee", "amount": 5.50},
        {"vendor": "AWS", "description": "cloud hosting", "amount": 89.00},
        {"vendor": "CVS Pharmacy", "description": "prescription", "amount": 35.00},
        {"vendor": "Staples", "description": "office supplies", "amount": 45.00},
        {"vendor": "Delta Airlines", "description": "flight ticket", "amount": 450.00},
        {"vendor": "Netflix", "description": "subscription", "amount": 15.99},
        {"vendor": "Amazon", "description": "printer ink", "amount": 25.00},
        {"vendor": "Verizon", "description": "phone bill", "amount": 85.00},
        {"vendor": "Coursera", "description": "online course", "amount": 49.00},
    ]
    
    print("\nTest Predictions:")
    print("-" * 70)
    print(f"{'Vendor':<20} {'Description':<15} {'Amount':<10} {'Category':<12} {'Source':<8} {'Conf':<6}")
    print("-" * 70)
    
    for test in test_cases:
        category, source, confidence = classifier.predict_category(
            test['vendor'],
            test['description'],
            test['amount']
        )
        
        print(f"{test['vendor']:<20} {test['description']:<15} "
              f"${test['amount']:<9.2f} {category:<12} {source:<8} {confidence:.2f}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


def main():
    """
    Main function for testing and demo.
    """
    print("\n" + "=" * 70)
    print("ML Model with Rule-Based Fallback")
    print("=" * 70 + "\n")
    
    # Run tests
    test_classifier()
    
    # Demo single prediction
    print("\nDemo: Single Prediction")
    print("-" * 70)
    result = classify_expense("Amazon", "printer ink", 25.00)
    print(f"Vendor: Amazon")
    print(f"Description: printer ink")
    print(f"Amount: $25.00")
    print(f"\nPrediction:")
    print(f"  Category: {result['category']}")
    print(f"  Source: {result['source']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Model Loaded: {result['model_loaded']}")
    
  

if __name__ == "__main__":
    main()
    




    








    