import os
import sys

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import should work
from src.data.preprocessing import load_data, clean_data, preprocess_features
from src.models.train import train_model

def main():
    # Load and preprocess data
    df = load_data('../../data/raw/car_data.csv')
    df_clean = clean_data(df)
    
    # Prepare features and target
    X, y, cat_cols, num_cols = preprocess_features(df_clean, target_column='Price')
    
    # Train and save model
    model, mse, r2 = train_model(X, y, cat_cols, num_cols)
    print(f"Model training completed with R² score: {r2:.4f}")

if __name__ == "__main__":
    main()
