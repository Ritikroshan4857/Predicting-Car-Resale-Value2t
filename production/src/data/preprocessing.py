import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path):
    """Load the car dataset from CSV file"""
    return pd.read_csv(file_path)
    
def clean_data(df):
    """Clean the data by handling missing values and outliers"""
    # Remove duplicate entries
    df = df.drop_duplicates()
    
    # Handle missing values
    # (Strategy depends on the specific dataset)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df
    
def preprocess_features(df, target_column='Price'):
    """Preprocess features for model training"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical features
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    
    # Create preprocessing pipelines (to be used in the model)
    return X, y, list(cat_cols), list(num_cols)
