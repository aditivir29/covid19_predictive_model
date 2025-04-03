import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os

def make_predictions():
    print("Starting prediction process...")
    
    # Check for model directory
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
        print("Created models directory")
    
    # Load model
    try:
        model = joblib.load("models/covid_model_improved.pkl")
        features = joblib.load("models/feature_names.pkl")
        print(f"Model loaded successfully with features: {features}")
    except FileNotFoundError:
        print("❌ Error: Model file not found. Please run train_model.py first.")
        return
    
    # Load latest data
    try:
        df = pd.read_csv("data/processed_data.csv", low_memory=False)
        print(f"Data loaded with {len(df)} rows")
    except FileNotFoundError:
        print("❌ Error: Processed data file not found. Please run preprocess.py first.")
        return
    
    # Convert date if it exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    
    # Get the most recent date in the data
    most_recent_date = df["Date"].max() if "Date" in df.columns else None
    print(f"Most recent date in data: {most_recent_date}")
    
    # For time-series predictions, we need to predict for the next day
    # We'll use the most recent data points for each state/region
    if "State/UnionTerritory" in df.columns:
        latest_data = df.sort_values("Date").groupby("State/UnionTerritory").last().reset_index()
    else:
        # If no state/territory column, just use the most recent data
        latest_data = df.sort_values("Date").tail(1)
        latest_data["State/UnionTerritory"] = "Overall"
    
    print(f"Using {len(latest_data)} latest data points for prediction")
    
    # Make sure all features are present
    missing_features = [f for f in features if f not in latest_data.columns]
    if missing_features:
        print(f"⚠️ Warning: Missing features: {missing_features}")
        # Add missing features with median values from full dataset
        for feature in missing_features:
            if feature in df.columns:
                latest_data[feature] = df[feature].median()
            else:
                latest_data[feature] = 0
                print(f"  Added default value 0 for {feature}")
    
    # Select only the required features in the correct order
    X_new = latest_data[features].copy()
    
    # Handle missing values and infinities
    X_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_new.fillna(X_new.median(), inplace=True)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    
    # Create prediction results dataframe
    results = pd.DataFrame({
        "State/UnionTerritory": latest_data["State/UnionTerritory"],
        "Predicted_Cases": predictions.round(0).astype(int)
    })
    
    # Calculate total predicted cases
    total_predicted = results["Predicted_Cases"].sum()
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Print predictions
    print("\n=== COVID-19 Prediction Results ===")
    print(f"Date of prediction: {datetime.now().strftime('%Y-%m-%d')}")
    if most_recent_date:
        next_date = most_recent_date + timedelta(days=1)
        print(f"Prediction for: {next_date.strftime('%Y-%m-%d')}")
    
    print(f"\n📈 Total Predicted COVID-19 Cases: {total_predicted:,}")
    print("\nPredictions by State/Union Territory:")
    
    # Sort by highest predicted cases
    for _, row in results.sort_values("Predicted_Cases", ascending=False).iterrows():
        if row["Predicted_Cases"] > 0:
            print(f"  {row['State/UnionTerritory']}: {row['Predicted_Cases']:,}")
    
    # Save predictions
    results.to_csv("output/predictions.csv", index=False)
    print("\n✅ Predictions saved to output/predictions.csv")

if __name__ == "__main__":
    make_predictions()