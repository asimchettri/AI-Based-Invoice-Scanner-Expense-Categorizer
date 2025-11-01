#this is the basic rule base classifier 

import logging
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class InvoiceClassifier:
  
    
    # Comprehensive category mappings
    CATEGORY_MAPPING = {
        'travel': {
            'keywords': [
                'airline', 'airways', 'flight', 'airport',
                'uber', 'lyft', 'taxi', 'cab', 'ride',
                'hotel', 'motel', 'resort', 'airbnb', 'booking',
                'hertz', 'avis', 'car rental', 'transport'
            ],
            'description': 'Travel and transportation'
        },
        'meals': {
            'keywords': [
                'restaurant', 'cafe', 'dine', 'diner',
                'pizza', 'burger', 'food', 'cuisine',
                'bakery', 'coffee', 'bar', 'pub',
                'delivery', 'grubhub', 'zomato', 'swiggy',
                'bistro', 'grill', 'kitchen'
            ],
            'description': 'Meals and restaurants'
        },
        'saas': {
            'keywords': [
                'stripe', 'paypal', 'subscription',
                'github', 'gitlab', 'aws', 'azure',
                'software', 'app', 'cloud', 'hosting',
                'database', 'api', 'service', 'platform',
                'heroku', 'digitalocean', 'linode',
                'slack', 'zoom', 'teams', 'confluence'
            ],
            'description': 'Software as a Service'
        },
        'office': {
            'keywords': [
                'stationery', 'office', 'supplies', 'supply',
                'paper', 'pen', 'desk', 'chair',
                'stapler', 'printer', 'ink', 'toner',
                'furniture', 'equipment'
            ],
            'description': 'Office supplies and equipment'
        },
        'utilities': {
            'keywords': [
                'electric', 'electricity', 'power',
                'water', 'gas', 'internet', 'telecom',
                'phone', 'mobile', 'broadband', 'wifi'
            ],
            'description': 'Utilities and communications'
        },
        'healthcare': {
            'keywords': [
                'pharmacy', 'hospital', 'clinic', 'doctor',
                'medical', 'medicine', 'health', 'dental',
                'pharmaceutical', 'healthcare'
            ],
            'description': 'Healthcare and medical'
        },
        'retail': {
            'keywords': [
                'store', 'shop', 'retail', 'walmart', 'target',
                'amazon', 'ebay', 'shopping', 'mart',
                'supermarket', 'grocery', 'market', 'laptop', 'smartphone',
                'product', 'item', 'equipment', 'gadget'
            ],
            'description': 'Retail and shopping'
        },
        'education': {
            'keywords': [
                'school', 'university', 'college', 'course',
                'training', 'education', 'academy', 'institute',
                'tuition', 'learning', 'class'
            ],
            'description': 'Education and training'
        },
        'entertainment': {
            'keywords': [
                'movie', 'cinema', 'theater', 'show',
                'concert', 'music', 'entertainment',
                'spotify', 'netflix', 'games', 'gaming',
                'streaming', 'ticket'
            ],
            'description': 'Entertainment and media'
        },
        'maintenance': {
            'keywords': [
                'repair', 'maintenance', 'cleaning', 'plumbing',
                'electrical', 'hvac', 'mechanic', 'service',
                'garage', 'workshop'
            ],
            'description': 'Maintenance and repairs'
        }
    }
    
    def __init__(self):
       
        logger.info("Invoice Classifier initialized successfully")
    
    def categorize_text(self, text: str) -> str:
       
        if not text:
            return 'other'
        
        text_lower = text.lower()
        
        # Check each category's keywords
        for category, data in self.CATEGORY_MAPPING.items():
            keywords = data['keywords']
            if any(keyword in text_lower for keyword in keywords):
                logger.info(f"Text categorized as '{category}'")
                return category
        
        logger.info("Text categorized as 'other'")
        return 'other'
    
    def categorize_invoice(self, vendor_name: str, invoice_text: Optional[str] = None) -> Dict:
       
        try:
            text_to_analyze = f"{vendor_name} {invoice_text or ''}".lower()
            
            # Score all categories
            scores = self._score_categories(text_to_analyze)
            
            # Get top category
            if scores:
                top_category = max(scores.items(), key=lambda x: x[1])
                category_name = top_category[0]
                confidence = top_category[1]
            else:
                category_name = 'other'
                confidence = 0.0
            
            result = {
                'category': category_name,
                'confidence': round(confidence, 2),
                'description': self.CATEGORY_MAPPING[category_name]['description']
                    if category_name in self.CATEGORY_MAPPING else 'Uncategorized',
                'all_scores': {k: round(v, 2) for k, v in scores.items()}
            }
            
            logger.info(f"Invoice categorized: {category_name} (confidence: {confidence})")
            return result
        
        except Exception as e:
            logger.error(f"Error categorizing invoice: {e}")
            return {
                'category': 'other',
                'confidence': 0.0,
                'description': 'Error in categorization',
                'error': str(e)
            }
    
    def _score_categories(self, text: str) -> Dict[str, float]:
       
        scores = {}
        
        for category, data in self.CATEGORY_MAPPING.items():
            keywords = data['keywords']
            matches = sum(1 for keyword in keywords if keyword in text)
            
            # Confidence score: matches / total keywords
            confidence = matches / len(keywords) if keywords else 0.0
            scores[category] = confidence
        
        return scores
    
    def get_categories(self) -> List[Dict]:
      
        return [
            {
                'name': name,
                'description': data['description'],
                'keyword_count': len(data['keywords'])
            }
            for name, data in self.CATEGORY_MAPPING.items()
        ]
    
    def add_category(self, category_name: str, keywords: List[str], description: str = '') -> bool:
      
        if category_name in self.CATEGORY_MAPPING:
            logger.warning(f"Category '{category_name}' already exists")
            return False
        
        self.CATEGORY_MAPPING[category_name] = {
            'keywords': [k.lower() for k in keywords],
            'description': description
        }
        
        logger.info(f"Category '{category_name}' added with {len(keywords)} keywords")
        return True
    
    def update_category_keywords(self, category_name: str, keywords: List[str], append: bool = False) -> bool:
       
        if category_name not in self.CATEGORY_MAPPING:
            logger.warning(f"Category '{category_name}' not found")
            return False
        
        new_keywords = [k.lower() for k in keywords]
        
        if append:
            existing = self.CATEGORY_MAPPING[category_name]['keywords']
            self.CATEGORY_MAPPING[category_name]['keywords'] = list(set(existing + new_keywords))
        else:
            self.CATEGORY_MAPPING[category_name]['keywords'] = new_keywords
        
        logger.info(f"Category '{category_name}' keywords updated")
        return True
    
    def get_category_stats(self) -> Dict:
       
        return {
            'total_categories': len(self.CATEGORY_MAPPING),
            'total_keywords': sum(len(data['keywords']) for data in self.CATEGORY_MAPPING.values()),
            'categories': {
                name: len(data['keywords'])
                for name, data in self.CATEGORY_MAPPING.items()
            }
        }

