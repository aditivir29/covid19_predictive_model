import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os

def train_model():
    print("Starting model training...")
    
    # Load dataset
    df = pd.read_csv("data/processed_data.csv", low_memory=False)
    
    # Convert date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    
    # Define Features & Target
    potential_features = [
        'Case Growth Rate', 'Testing Positivity Rate', 
        'Youth_Population_Pct', 'Women_Tobacco_Use_Pct',
        'Vaccination Effectiveness', 'Month', 'Week', 
        'Day_of_week', 'Days_since_start'
    ]
    
    # Use only features that exist in the DataFrame
    features = [f for f in potential_features if f in df.columns]
    target = 'New_Cases'  # Predict new cases instead of cumulative confirmed cases
    
    if target not in df.columns:
        print(f"Target column '{target}' not found. Using 'Confirmed' instead.")
        target = 'Confirmed'
    
    print(f"Using features: {features}")
    print(f"Target: {target}")
    
    # Prepare features and target
    X = df[features].copy()
    y = df[target]
    
    # Clean data
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a robust pipeline
    pipeline = Pipeline([
        ('scaler', RobustScaler()),  # RobustScaler is less influenced by outliers
        ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
    ])
    
    # Fit the model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_test_pred = pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    # Feature importance analysis
    importances = pipeline.named_steps['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create directory for outputs if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances for COVID-19 Prediction")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig("output/feature_importance.png")
    
    # Print feature importances
    print("\nFeature Importances:")
    for i in range(X.shape[1]):
        print(f"{features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model and feature names
    joblib.dump(pipeline, "models/covid_model_improved.pkl")
    joblib.dump(features, "models/feature_names.pkl")
    
    print("✅ Improved Model Trained & Saved to models/covid_model_improved.pkl")

if __name__ == "__main__":
    train_model()