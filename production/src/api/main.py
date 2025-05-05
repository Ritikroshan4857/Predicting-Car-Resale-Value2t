from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predict import load_model, predict_price

# Load the model
model = load_model()

# Create the FastAPI app
app = FastAPI(
    title="Car Resale Value Prediction API",
    description="API for predicting the resale value of cars based on their features",
    version="1.0"
)

# Define the input data model
class CarFeatures(BaseModel):
    make: str
    model: str
    year: int
    mileage: float
    fuel_type: str
    # Add other features as needed based on your model
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Resale Value Prediction API"}

@app.post("/predict/")
def predict_car_price(car: CarFeatures):
    try:
        # Convert input to DataFrame
        features = pd.DataFrame([car.dict()])
        
        # Make prediction
        price = predict_price(model, features)
        
        return {
            "predicted_price": round(price, 2),
            "car_details": car.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
