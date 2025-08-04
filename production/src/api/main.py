from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import sys
import os
import time
import hashlib
import json
from typing import Optional, Dict, Any
from functools import lru_cache
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predict import load_model, predict_price, ModelPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance with lazy loading
_model_predictor: Optional[ModelPredictor] = None

def get_model_predictor() -> ModelPredictor:
    """Lazy load the model predictor singleton"""
    global _model_predictor
    if _model_predictor is None:
        logger.info("Loading model predictor...")
        _model_predictor = ModelPredictor()
        logger.info("Model predictor loaded successfully")
    return _model_predictor

# Create the FastAPI app with optimizations
app = FastAPI(
    title="Car Resale Value Prediction API",
    description="Optimized API for predicting the resale value of cars based on their features",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware for performance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Define the input data model with validation
class CarFeatures(BaseModel):
    make: str = Field(..., min_length=1, max_length=50, description="Car manufacturer")
    model: str = Field(..., min_length=1, max_length=50, description="Car model")
    year: int = Field(..., ge=1900, le=2030, description="Manufacturing year")
    mileage: float = Field(..., ge=0, le=1000000, description="Car mileage")
    fuel_type: str = Field(..., min_length=1, max_length=20, description="Fuel type")
    
    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            np.integer: lambda v: int(v),
            np.floating: lambda v: float(v),
        }

# Response model for better API documentation
class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_score: Optional[float] = None
    car_details: Dict[str, Any]
    processing_time_ms: float

# Cache for predictions (in-memory cache for demo, use Redis in production)
_prediction_cache: Dict[str, Dict[str, Any]] = {}

def generate_cache_key(car_features: CarFeatures) -> str:
    """Generate a cache key for the prediction"""
    feature_str = json.dumps(car_features.dict(), sort_keys=True)
    return hashlib.md5(feature_str.encode()).hexdigest()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {
        "message": "Welcome to the Optimized Car Resale Value Prediction API",
        "version": "2.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        predictor = get_model_predictor()
        return {
            "status": "healthy",
            "model_loaded": predictor.is_loaded(),
            "cache_size": len(_prediction_cache)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict/", response_model=PredictionResponse)
async def predict_car_price(car: CarFeatures):
    """Predict car price with caching and performance optimizations"""
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = generate_cache_key(car)
        if cache_key in _prediction_cache:
            cached_result = _prediction_cache[cache_key]
            cached_result["processing_time_ms"] = (time.time() - start_time) * 1000
            return cached_result
        
        # Get model predictor
        predictor = get_model_predictor()
        
        # Convert input to optimized format
        features_dict = car.dict()
        features_df = pd.DataFrame([features_dict])
        
        # Make prediction
        prediction_result = predictor.predict(features_df)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Prepare response
        response_data = {
            "predicted_price": round(prediction_result["price"], 2),
            "confidence_score": prediction_result.get("confidence", None),
            "car_details": features_dict,
            "processing_time_ms": round(processing_time, 2)
        }
        
        # Cache the result (limit cache size to prevent memory issues)
        if len(_prediction_cache) < 1000:  # Limit cache to 1000 entries
            _prediction_cache[cache_key] = response_data.copy()
        
        return response_data
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.delete("/cache/clear")
async def clear_cache():
    """Clear the prediction cache"""
    global _prediction_cache
    cache_size = len(_prediction_cache)
    _prediction_cache.clear()
    return {"message": f"Cache cleared. Removed {cache_size} entries"}

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "cache_size": len(_prediction_cache),
        "cache_keys": list(_prediction_cache.keys())[:10]  # Show first 10 keys
    }

# Error handlers for better performance
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error responses"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"}
    )
