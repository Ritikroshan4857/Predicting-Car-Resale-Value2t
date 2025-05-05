import joblib
import pandas as pd

def load_model(model_path='../../models/car_price_model.joblib'):
    """Load the trained model from disk"""
    return joblib.load(model_path)

def predict_price(model, features):
    """Predict car price based on input features"""
    # Convert input features to DataFrame if not already
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(features)
    
    return prediction[0]
