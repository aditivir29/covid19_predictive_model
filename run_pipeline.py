import os

print("🚀 Running COVID-19 Predictive Model Pipeline...")

# Run Data Preprocessing
os.system("python src/preprocess.py")

# Run EDA
os.system("python src/eda.py")

# Train Model
os.system("python src/train_model.py")

# Make Predictions
os.system("python src/predict.py")

print("✅ Pipeline Execution Complete!")
