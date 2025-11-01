import joblib
import sklearn
import numpy as np

print("Current NumPy version:", np.__version__)
print("Current scikit-learn version:", sklearn.__version__)

try:
    # Try to load the model and see the error
    model = joblib.load('models/classifier.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    
    # Try to get more info about the model
    try:
        with open('models/classifier.joblib', 'rb') as f:
            import pickle
            # This might reveal more about the model structure
            data = pickle.load(f)
            print("Model info:", type(data))
    except Exception as e2:
        print("Detailed error:", e2)