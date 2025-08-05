import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
import time
import os

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Optimized model predictor with lazy loading and caching"""
    
    def __init__(self, model_path: str = '../../models/car_price_model.joblib'):
        self.model_path = model_path
        self._model = None
        self._preprocessor = None
        self._is_loaded = False
        self._load_time = None
        self._prediction_cache = {}
        
    def _load_model(self):
        """Lazy load the model with error handling"""
        if self._is_loaded:
            return
            
        start_time = time.time()
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load the model
            loaded_model = joblib.load(self.model_path)
            
            # Extract preprocessor and model components
            if hasattr(loaded_model, 'named_steps'):
                self._preprocessor = loaded_model.named_steps.get('preprocessor')
                self._model = loaded_model.named_steps.get('regressor')
            else:
                # Fallback for different model formats
                self._model = loaded_model
                
            self._is_loaded = True
            self._load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {self._load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded
    
    def get_load_time(self) -> Optional[float]:
        """Get model load time"""
        return self._load_time
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with caching and optimization"""
        # Ensure model is loaded
        self._load_model()
        
        # Generate cache key
        cache_key = self._generate_cache_key(features)
        
        # Check cache first
        if cache_key in self._prediction_cache:
            logger.debug("Returning cached prediction")
            return self._prediction_cache[cache_key]
        
        try:
            start_time = time.time()
            
            # Ensure features are in the correct format
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame([features])
            
            # Make prediction
            if self._preprocessor is not None:
                # Use the full pipeline
                prediction = self._model.predict(features)
            else:
                # Direct prediction
                prediction = self._model.predict(features)
            
            # Calculate confidence score (if available)
            confidence = self._calculate_confidence(prediction, features)
            
            prediction_time = time.time() - start_time
            
            result = {
                "price": float(prediction[0]),
                "confidence": confidence,
                "prediction_time": prediction_time
            }
            
            # Cache the result (limit cache size)
            if len(self._prediction_cache) < 500:
                self._prediction_cache[cache_key] = result
            
            logger.debug(f"Prediction completed in {prediction_time:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction error: {e}")
    
    def _generate_cache_key(self, features: pd.DataFrame) -> str:
        """Generate cache key for features"""
        # Convert to string representation for hashing
        feature_str = features.to_string()
        return str(hash(feature_str))
    
    def _calculate_confidence(self, prediction: np.ndarray, features: pd.DataFrame) -> Optional[float]:
        """Calculate confidence score if model supports it"""
        try:
            # For Random Forest, we can use prediction variance
            if hasattr(self._model, 'estimators_'):
                predictions = []
                for estimator in self._model.estimators_:
                    if self._preprocessor is not None:
                        # Apply preprocessing
                        processed_features = self._preprocessor.transform(features)
                        pred = estimator.predict(processed_features)
                    else:
                        pred = estimator.predict(features)
                    predictions.append(pred[0])
                
                # Calculate confidence based on variance
                predictions_array = np.array(predictions)
                confidence = 1.0 / (1.0 + np.std(predictions_array))
                return float(confidence)
            
            return None
        except Exception:
            return None
    
    def clear_cache(self):
        """Clear the prediction cache"""
        cache_size = len(self._prediction_cache)
        self._prediction_cache.clear()
        logger.info(f"Cleared prediction cache ({cache_size} entries)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._prediction_cache),
            "model_loaded": self._is_loaded,
            "load_time": self._load_time
        }

# Legacy functions for backward compatibility
def load_model(model_path='../../models/car_price_model.joblib'):
    """Load the trained model from disk (legacy function)"""
    predictor = ModelPredictor(model_path)
    predictor._load_model()
    return predictor._model

def predict_price(model, features):
    """Predict car price based on input features (legacy function)"""
    predictor = ModelPredictor()
    predictor._model = model
    result = predictor.predict(features)
    return result["price"]
