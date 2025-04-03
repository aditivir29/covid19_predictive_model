# COVID-19 Predictive Model: Data Pipeline for Indian States and Union Territories

## Overview
This project implements a machine learning-based approach to predict COVID-19 case numbers across Indian states and union territories using historical data, demographic information, and calculated health metrics. The data pipeline includes preprocessing, exploratory data analysis, model training, and prediction generation.

## Project Structure
```
COVID19_PREDICTIVE_MODEL/
├── .venv/  # Python virtual environment
├── data/
│   ├── covid_19_india.csv  # Raw COVID-19 data for India
│   ├── covid_vaccine_statewise.csv  # Vaccination data by state
│   ├── datafile.csv  # Health and demographic data
│   ├── predicted_vs_actual.csv  # Comparison of predictions with actual data
│   ├── processed_data.csv  # Cleaned and feature-engineered data
│   ├── StatewiseTestingDetails.csv  # Testing data by state
├── env/  # Environment configuration
├── models/
│   ├── covid_model_improved.pkl  # Improved prediction model
│   ├── covid_model.pkl  # Base prediction model
│   └── feature_names.pkl  # Model feature names
├── output/
│   ├── case_growth_trend.png
│   ├── cases_by_state.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── positivity_vs_growth.png
│   ├── predictions.csv  # Generated predictions
│   └── testing_positivity_distribution.png
├── reports/
│   └── Social Impact Report on COVID.pdf  # Analysis report
├── src/
│   ├── eda.py  # Exploratory data analysis and visualization
│   ├── predict.py  # Generating future predictions
│   ├── preprocess.py  # Data cleaning and feature engineering
│   ├── train_model.py  # Model training and evaluation
│   └── run_pipeline.py  # Script to run the complete pipeline
├── .gitignore
├── README.md
└── requirements.txt  # Project dependencies
```

## Features
### Data Preprocessing
- Cleans raw data from `covid_19_india.csv`, `StatewiseTestingDetails.csv`, `covid_vaccine_statewise.csv`, and `datafile.csv`
- Handles missing values
- Engineers features such as case growth rate and testing positivity rate

### Exploratory Data Analysis
- Generates visualizations to understand COVID-19 trends and patterns

### Model Training
- Implements a machine learning model to predict new COVID-19 cases

### Prediction Generation
- Forecasts future COVID-19 cases for Indian states and union territories

## Key Features Used in Modeling
- **Case Growth Rate** (from `covid_19_india.csv`)
- **Testing Positivity Rate** (from `StatewiseTestingDetails.csv`)
- **Demographic Information** (from `datafile.csv`, e.g., Youth Population %, Women Tobacco Use %)
- **Vaccination Effectiveness Estimates** (from `covid_vaccine_statewise.csv`)
- **Temporal Features** (Month, Week, Day of week, Days since pandemic start)

## Usage
### Prerequisites
- Python 3.6+
- Dependencies listed in `requirements.txt`

### Installation
Clone the repository:
```
git clone <repository-url>
cd COVID19_PREDICTIVE_MODEL
```
Create and activate a virtual environment:
```
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```

### Running the Pipeline
Run individual components:
```
python src/preprocess.py   # Data Preprocessing
python src/eda.py          # Exploratory Data Analysis
python src/train_model.py  # Model Training
python src/predict.py      # Generate Predictions
```
Or run the entire pipeline:
```
python src/run_pipeline.py
```

## Model Performance
- **Test RMSE:** 23.41
- **Test R²:** 0.9997
- **Test MAE:** 1.46

### Feature Importance
- **Testing Positivity Rate:** 90.11%
- **Case Growth Rate:** 8.60%
- **Days Since Pandemic Start:** 1.25%
- **Other Features:** <0.1% each

## Visualizations
The project generates multiple visual insights:
- Case growth trends over time
- Cases by state comparison
- Feature correlation heatmap
- Feature importance chart
- Positivity rate vs. growth rate analysis
- Testing positivity distribution

## Social Impact Report
A comprehensive **Social Impact Report** (available in the `reports/` folder) covers:
- Societal impact of COVID-19 across Indian states and demographics
- Role of predictive modeling in public health decision-making
- Suggested interventions based on predictive insights
- Recommendations for resource allocation in high-risk areas
- Long-term implications for health infrastructure planning

## Notes
- Uses the latest 46 data points for forecasting
- Predicts state-wise COVID-19 cases for upcoming days
- Estimates testing positivity rate when direct data is unavailable

## Future Improvements
- Integration of more detailed vaccination data
- Incorporation of mobility data to capture population movement effects
- Implementation of advanced time series models
- Development of regional sub-models for targeted predictions

