# ml_model.py - Model loader with rule-based fallback

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import joblib
import warnings

# Suppress warnings during model loading
warnings.filterwarnings('ignore', category=UserWarning)

# Check dependency versions at startup
try:
    import numpy as np
    import sklearn
    NUMPY_VERSION = np.__version__
    SKLEARN_VERSION = sklearn.__version__
except ImportError as e:
    print(f"❌ Critical dependency missing: {e}")
    sys.exit(1)

# Import existing classifier
try:
    from classifier import InvoiceClassifier
    RULE_CLASSIFIER_AVAILABLE = True 
except ImportError:
    print("⚠️  Warning: classifier.py not found. Rule-based classification will be limited.")
    RULE_CLASSIFIER_AVAILABLE = False
    InvoiceClassifier = None

# Import text cleaning utilities
try:
    from utils.cleaner import clean_text_for_model, extract_amounts, extract_total_amount
except ImportError:
    print("⚠️  Warning: cleaner.py not found. Using basic text cleaning")
    def clean_text_for_model(text, lowercase=True):
        return text.lower().strip() if lowercase else text.strip()
    def extract_amounts(text):
        import re
        amounts = re.findall(r'\d+\.?\d{0,2}', text)
        return [float(a) for a in amounts if a]
    def extract_total_amount(text):
        amounts = extract_amounts(text)
        return max(amounts) if amounts else None

# Configure logging
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


def check_version_compatibility():
    """Check if NumPy and scikit-learn versions are compatible"""
    try:
        sklearn_version = tuple(map(int, SKLEARN_VERSION.split('.')[:2]))
        numpy_version = tuple(map(int, NUMPY_VERSION.split('.')[:2]))
        
        # sklearn 1.7+ requires numpy 2.0+
        if sklearn_version >= (1, 7) and numpy_version < (2, 0):
            logger.error(
                f"❌ Version mismatch: scikit-learn {SKLEARN_VERSION} requires NumPy 2.0+, "
                f"but found NumPy {NUMPY_VERSION}"
            )
            return False
        
        logger.info(f"✅ Dependencies compatible: NumPy {NUMPY_VERSION}, scikit-learn {SKLEARN_VERSION}")
        return True
        
    except Exception as e:
        logger.warning(f"Could not verify version compatibility: {e}")
        return True  # Proceed anyway


class ExpenseClassifier:
    """
    Expense classifier with ML model and rule-based fallback.
    Automatically falls back to rule-based classification if ML model fails.
    """

    def __init__(self, model_path: str = "models/classifier.joblib"):
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = None
        self.rule_classifier = None
        self.model_loaded = False
        
        # Check version compatibility
        if not check_version_compatibility():
            logger.warning("Version incompatibility detected. ML model may not load correctly.")
        
        # Try to load ML model
        self._load_model()

        # Initialize rule-based classifier as fallback
        if RULE_CLASSIFIER_AVAILABLE:
            try:
                self.rule_classifier = InvoiceClassifier()
                logger.info("✅ Rule-based classifier initialized as fallback")
            except Exception as e:
                logger.error(f"❌ Error initializing rule-based classifier: {e}")
        
        logger.info(f"ExpenseClassifier initialized (ML Model: {self.model_loaded})")

    def _load_model(self) -> bool:
        """Load the trained ML model with comprehensive error handling"""
        
        try:
            if not self.model_path.exists():
                logger.warning(f"⚠️  Model file not found at: {self.model_path}")
                logger.info("Will use rule-based classification as fallback")
                return False
            
            # Check file size
            file_size = self.model_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is suspicious
                logger.warning(f"⚠️  Model file seems too small ({file_size} bytes). May be corrupted.")
                return False
            
            logger.info(f"Loading model from: {self.model_path} ({file_size / (1024*1024):.2f} MB)")
            
            # Load the model
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ ML model loaded successfully")
            logger.info(f"   Model type: {type(self.model).__name__}")
            
            # Verify model has required methods
            if not hasattr(self.model, 'predict'):
                logger.error("❌ Loaded model doesn't have 'predict' method")
                self.model = None
                return False
            
            # Load metadata
            metadata_path = self.model_path.parent / "metadata.joblib"
            if metadata_path.exists():
                try:
                    self.metadata = joblib.load(metadata_path)
                    logger.info(f"✅ Metadata loaded successfully")
                    logger.info(f"   Model name: {self.metadata.get('model_name', 'Unknown')}")
                    logger.info(f"   Accuracy: {self.metadata.get('accuracy', 0)*100:.2f}%")
                    logger.info(f"   Trained at: {self.metadata.get('trained_at', 'Unknown')}")
                    logger.info(f"   Training samples: {self.metadata.get('training_samples', 'Unknown')}")
                    logger.info(f"   NumPy version (training): {self.metadata.get('numpy_version', 'Unknown')}")
                    logger.info(f"   Scikit-learn version (training): {self.metadata.get('sklearn_version', 'Unknown')}")
                    
                    # Check for version mismatch
                    if self.metadata.get('numpy_version') != NUMPY_VERSION:
                        logger.warning(
                            f"⚠️  NumPy version mismatch: "
                            f"Model trained with {self.metadata.get('numpy_version')}, "
                            f"running with {NUMPY_VERSION}"
                        )
                    
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
                    self.metadata = self._get_default_metadata()
            else:
                logger.warning("⚠️  Metadata file not found, using defaults")
                self.metadata = self._get_default_metadata()
            
            self.model_loaded = True
            return True
            
        except ModuleNotFoundError as e:
            logger.error(f"❌ Module not found error: {e}")
            logger.error("   This usually means NumPy/scikit-learn version mismatch")
            logger.error(f"   Current versions: NumPy {NUMPY_VERSION}, scikit-learn {SKLEARN_VERSION}")
            self.model = None
            self.model_loaded = False
            return False
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {type(e).__name__}: {str(e)}")
            logger.info("Will use rule-based classification as fallback")
            self.model = None
            self.model_loaded = False
            return False
    
    def _get_default_metadata(self) -> Dict:
        """Return default metadata when file is missing"""
        return {
            'model_name': type(self.model).__name__ if self.model else 'Unknown',
            'accuracy': 0.85,
            'categories': CATEGORIES,
            'trained_at': 'Unknown',
            'training_samples': 'Unknown'
        }
    
    def rule_based_classify(self, text: str) -> Tuple[str, float]:
        """
        Perform rule-based classification as fallback.
        Returns: (category, confidence)
        """
        
        # Use advanced rule-based classifier if available
        if self.rule_classifier:
            try:
                result = self.rule_classifier.categorize_invoice(
                    vendor_name=text,
                    invoice_text=text
                )
                return result['category'], result['confidence']
            except Exception as e:
                logger.warning(f"Rule classifier failed: {e}, using simple rules")
        
        # Simple keyword-based rules
        text_lower = text.lower()
        
        simple_rules = {
            'travel': ['uber', 'lyft', 'taxi', 'airline', 'flight', 'hotel', 'airbnb', 'rental car'],
            'meals': ['restaurant', 'pizza', 'burger', 'starbucks', 'coffee', 'food', 'dining', 'cafe'],
            'saas': ['aws', 'github', 'slack', 'zoom', 'subscription', 'software', 'cloud', 'hosting'],
            'office': ['staples', 'office', 'supplies', 'printer', 'paper', 'depot'],
            'utilities': ['electric', 'power', 'internet', 'phone', 'verizon', 'at&t', 'comcast', 'utility'],
            'healthcare': ['pharmacy', 'cvs', 'walgreens', 'medical', 'hospital', 'doctor', 'health'],
            'retail': ['amazon', 'walmart', 'target', 'store', 'shopping', 'costco'],
            'education': ['course', 'training', 'udemy', 'coursera', 'school', 'university', 'tuition'],
            'entertainment': ['netflix', 'spotify', 'hulu', 'movie', 'music', 'theater', 'concert'],
            'maintenance': ['repair', 'maintenance', 'plumbing', 'cleaning', 'fix', 'service']
        }
        
        # Check each category
        for category, keywords in simple_rules.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category, 0.7  # Medium confidence for keyword match
        
        return 'other', 0.5  # Low confidence for default

    def predict_category(
        self, 
        vendor: str, 
        description: str = "", 
        amount: float = 0.0
    ) -> Tuple[str, str, float]:
        """
        Predict expense category using ML model or fallback to rules.
        
        Args:
            vendor: Vendor name
            description: Transaction description
            amount: Transaction amount
            
        Returns:
            (category, source, confidence)
            - category: predicted category
            - source: 'model' or 'rule'
            - confidence: 0.0-1.0
        """
        
        # Clean and combine features
        vendor_clean = clean_text_for_model(str(vendor)) if vendor else ""
        description_clean = clean_text_for_model(str(description)) if description else ""
        amount_str = f"amount_{int(amount)}" if amount > 0 else "amount_0"
        
        combined_text = f"{vendor_clean} {description_clean} {amount_str}"
        
        # Try ML model first
        if self.model_loaded and self.model:
            try:
                prediction = self.model.predict([combined_text])[0]
                
                # Get confidence if model supports predict_proba
                try:
                    proba = self.model.predict_proba([combined_text])[0]
                    confidence = float(max(proba))
                except AttributeError:
                    confidence = 0.85  # Default confidence for models without proba
                
                logger.debug(f"ML Model prediction: {prediction} (confidence: {confidence:.2f})")
                return prediction, "model", confidence
            
            except Exception as e:
                logger.error(f"❌ Error during ML prediction: {str(e)}")
                logger.info("Falling back to rule-based classification")
        
        # Fallback to rule-based
        category, confidence = self.rule_based_classify(combined_text)
        logger.debug(f"Rule-based prediction: {category} (confidence: {confidence:.2f})")
        return category, "rule", confidence
    
    def predict_from_text(self, text: str) -> Tuple[str, str, float]:
        """
        Predict category from raw text.
        
        Args:
            text: Raw text to classify
            
        Returns:
            (category, source, confidence)
        """
        
        if self.model_loaded and self.model:
            try:
                cleaned = clean_text_for_model(text)
                prediction = self.model.predict([cleaned])[0]
                
                try:
                    proba = self.model.predict_proba([cleaned])[0]
                    confidence = float(max(proba))
                except AttributeError:
                    confidence = 0.85
                
                return prediction, "model", confidence
            
            except Exception as e:
                logger.error(f"Error during ML prediction: {str(e)}")
        
        # Fallback to rules
        category, confidence = self.rule_based_classify(text)
        return category, "rule", confidence
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        
        info = {
            'model_loaded': self.model_loaded,
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'rule_classifier_available': self.rule_classifier is not None,
            'categories': CATEGORIES,
            'numpy_version': NUMPY_VERSION,
            'sklearn_version': SKLEARN_VERSION
        }
        
        if self.metadata:
            info.update({
                'model_name': self.metadata.get('model_name'),
                'accuracy': self.metadata.get('accuracy'),
                'trained_at': self.metadata.get('trained_at'),
                'training_samples': self.metadata.get('training_samples'),
                'data_source': self.metadata.get('data_source'),
                'training_numpy_version': self.metadata.get('numpy_version'),
                'training_sklearn_version': self.metadata.get('sklearn_version')
            })
        
        return info
    
    def reload_model(self) -> bool:
        """Reload the model from disk"""
        logger.info("Reloading model...")
        self.model_loaded = False
        self.model = None
        self.metadata = None
        return self._load_model()


# Helper functions for amount extraction
def extract_amount_from_text(text: str) -> Optional[float]:
    """Extract the primary amount from text"""
    return extract_total_amount(text)


def extract_all_amounts(text: str) -> list:
    """Extract all amounts found in text"""
    return extract_amounts(text)


def classify_expense(vendor: str, description: str = "", amount: float = 0.0) -> Dict:
    """
    Convenience function to classify an expense.
    Creates a new classifier instance each time.
    """
    classifier = ExpenseClassifier()
    category, source, confidence = classifier.predict_category(vendor, description, amount)
    
    return {
        'category': category,
        'source': source,
        'confidence': confidence,
        'model_loaded': classifier.model_loaded
    }


# Test function
def test_classifier():
    """Test the expense classifier with sample data"""
    
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
    print("✅ Test complete!")
    print("=" * 70)


def main():
    """Main function for testing and demo"""
    
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