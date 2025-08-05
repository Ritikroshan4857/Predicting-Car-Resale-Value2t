import pandas as pd
import numpy as np
import joblib
import time
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
import warnings
import os
from typing import Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

class OptimizedModelTrainer:
    """Optimized model trainer with performance monitoring and model compression"""
    
    def __init__(self, model_path: str = '../../models/car_price_model.joblib'):
        self.model_path = model_path
        self.best_model = None
        self.training_time = None
        self.cv_scores = None
        
    def build_model(self, cat_cols: list, num_cols: list) -> Pipeline:
        """Build the preprocessing and model pipeline with optimizations"""
        
        # Optimized preprocessing for numerical data
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Optimized preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ],
            remainder='drop'  # Drop any remaining columns
        )
        
        # Create the modeling pipeline with optimized Random Forest
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                random_state=42,
                n_jobs=-1,  # Use all CPU cores
                bootstrap=True,
                oob_score=True  # Out-of-bag scoring for better validation
            ))
        ])
        
        return model
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, cat_cols: list, num_cols: list) -> Dict[str, Any]:
        """Train the model with optimizations and return comprehensive results"""
        
        start_time = time.time()
        logger.info("Starting model training...")
        
        try:
            # Split the data with stratification if possible
            try:
                # Try stratified split if target has enough unique values
                if len(y.unique()) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=10, duplicates='drop')
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
            except:
                # Fallback to regular split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Build the model pipeline
            model = self.build_model(cat_cols, num_cols)
            
            # Optimized hyperparameter grid
            param_grid = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [None, 15, 20, 25],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
                'regressor__max_features': ['sqrt', 'log2', None]
            }
            
            logger.info("Starting hyperparameter tuning...")
            
            # Perform grid search with optimized settings
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=5, 
                scoring='r2',
                n_jobs=-1,  # Use all CPU cores
                verbose=1,
                return_train_score=True
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            self.best_model = grid_search.best_estimator_
            
            # Cross-validation scores
            self.cv_scores = cross_val_score(
                self.best_model, X_train, y_train, cv=5, scoring='r2'
            )
            
            # Evaluate on test set
            y_pred = self.best_model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate feature importance if available
            feature_importance = self._get_feature_importance(X, cat_cols, num_cols)
            
            # Training time
            self.training_time = time.time() - start_time
            
            # Compile results
            results = {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'cv_scores_mean': self.cv_scores.mean(),
                'cv_scores_std': self.cv_scores.std(),
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'training_time': self.training_time,
                'feature_importance': feature_importance,
                'model_size_mb': self._get_model_size()
            }
            
            # Log results
            logger.info(f"Training completed in {self.training_time:.2f}s")
            logger.info(f"Best CV Score: {grid_search.best_score_:.4f}")
            logger.info(f"Test R²: {test_r2:.4f}")
            logger.info(f"Test RMSE: {test_rmse:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}")
    
    def _get_feature_importance(self, X: pd.DataFrame, cat_cols: list, num_cols: list) -> Dict[str, float]:
        """Extract feature importance from the model"""
        try:
            if hasattr(self.best_model.named_steps['regressor'], 'feature_importances_'):
                # Get feature names after preprocessing
                feature_names = []
                
                # Add numerical feature names
                feature_names.extend(num_cols)
                
                # Add categorical feature names (after one-hot encoding)
                if cat_cols:
                    preprocessor = self.best_model.named_steps['preprocessor']
                    cat_transformer = preprocessor.named_transformers_['cat']
                    if hasattr(cat_transformer, 'get_feature_names_out'):
                        cat_feature_names = cat_transformer.get_feature_names_out(cat_cols)
                        feature_names.extend(cat_feature_names)
                
                # Create importance dictionary
                importances = self.best_model.named_steps['regressor'].feature_importances_
                feature_importance = dict(zip(feature_names, importances))
                
                # Sort by importance
                return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return {}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    def _get_model_size(self) -> float:
        """Calculate model size in MB"""
        try:
            if self.best_model is not None:
                # Save model temporarily to check size
                temp_path = f"{self.model_path}.temp"
                joblib.dump(self.best_model, temp_path, compress=3)
                size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                os.remove(temp_path)
                return size_mb
            return 0.0
        except Exception:
            return 0.0
    
    def save_model(self, compress: bool = True) -> str:
        """Save the trained model with optional compression"""
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save with compression
            if compress:
                joblib.dump(self.best_model, self.model_path, compress=3)
            else:
                joblib.dump(self.best_model, self.model_path)
            
            logger.info(f"Model saved to {self.model_path}")
            return self.model_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Model saving failed: {e}")

# Legacy functions for backward compatibility
def build_model(cat_cols, num_cols):
    """Build the preprocessing and model pipeline (legacy function)"""
    trainer = OptimizedModelTrainer()
    return trainer.build_model(cat_cols, num_cols)

def train_model(X, y, cat_cols, num_cols, model_path='../../models/car_price_model.joblib'):
    """Train the model and save it to disk (legacy function)"""
    trainer = OptimizedModelTrainer(model_path)
    results = trainer.train_model(X, y, cat_cols, num_cols)
    trainer.save_model()
    
    print(f"Best Model Parameters: {results['best_params']}")
    print(f"Test MSE: {results['test_mse']:.2f}")
    print(f"Test R²: {results['test_r2']:.2f}")
    print(f"Training Time: {results['training_time']:.2f}s")
    
    return trainer.best_model, results['test_mse'], results['test_r2']
