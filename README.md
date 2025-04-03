# COVID-19 Predictive Model

A comprehensive data pipeline for preprocessing, analyzing, and forecasting COVID-19 cases across Indian states and union territories.

## Overview

This project implements a machine learning-based approach to predict COVID-19 case numbers using historical data, demographic information, and calculated health metrics. The pipeline includes data preprocessing, exploratory data analysis, model training, and prediction generation.

## Project Structure

COVID19_PREDICTIVE_MODEL/
├── .venv/                      # Python virtual environment
├── data/
│   ├── covid_19_india.csv      # Raw COVID-19 data for India
│   ├── covid_vaccine_statewise.csv  # Vaccination data by state
│   ├── datafile.csv            # Health and demographic data
│   ├── predicted_vs_actual.csv # Comparison of predictions with actual data
│   ├── processed_data.csv      # Cleaned and feature-engineered data
│   └── StatewiseTestingDetails.csv  # Testing data by state
├── env/                        # Environment configuration
│   ├── etc/
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── pyvenv.cfg
├── models/
│   ├── covid_model_improved.pkl  # Improved prediction model
│   ├── covid_model.pkl          # Base prediction model
│   └── feature_names.pkl        # Model feature names
├── output/
│   ├── case_growth_trend.png
│   ├── cases_by_state.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── positivity_vs_growth.png
│   ├── predictions.csv          # Generated predictions
│   └── testing_positivity_distribution.png
├── reports/
│   └── Social Impact Report on COVID.pdf  # Analysis report
├── src/
│   ├── eda.py                  # Exploratory data analysis and visualization
│   ├── predict.py              # Generating future predictions
│   ├── preprocess.py           # Data cleaning and feature engineering
│   ├── train_model.py          # Model training and evaluation
│   └── run_pipeline.py         # Script to run the complete pipeline
├── .gitignore
├── README.md
└── requirements.txt           # Project dependencies


## Features

- **Data Preprocessing**: Cleans raw data, handles missing values, and engineers features such as case growth rate and testing positivity rate
- **Exploratory Data Analysis**: Generates visualizations to understand COVID-19 trends and patterns
- **Model Training**: Implements a machine learning model to predict new COVID-19 cases
- **Prediction Generation**: Forecasts future COVID-19 cases by state/union territory

## Key Features Used in Modeling

- Case Growth Rate
- Testing Positivity Rate
- Demographic information (Youth Population %, Women Tobacco Use %)
- Vaccination Effectiveness estimates
- Temporal features (Month, Week, Day of week, Days since pandemic start)

## Usage

### Prerequisites

- Python 3.6+
- Dependencies listed in requirements.txt

### Installation

1. Clone the repository
2. Create a virtual environment:

   python -m venv .venv

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Install dependencies:
   
   pip install -r requirements.txt
   

### Running the Pipeline

You can run individual components:

1. **Data Preprocessing**:
   
   python src/preprocess.py
   

2. **Exploratory Data Analysis**:
   
   python src/eda.py
   

3. **Model Training**:
   
   python src/train_model.py
   

4. **Generate Predictions**:
   
   python src/predict.py
   

Or run the entire pipeline:

python src/run_pipeline.py


## Model Performance

The current model achieves impressive predictive performance:
- Test RMSE: 23.41
- Test R²: 0.9997
- Test MAE: 1.46

## Feature Importance

Based on our model analysis, the key drivers of COVID-19 case predictions are:
1. Testing Positivity Rate (90.11%)
2. Case Growth Rate (8.60%)
3. Days since pandemic start (1.25%)
4. Other features (< 0.1% each)

## Visualizations

The project generates several visualizations to help understand COVID-19 patterns:
- Case growth trends over time
- Cases by state comparison
- Feature correlation heatmap
- Feature importance chart
- Positivity rate vs. growth rate analysis
- Testing positivity distribution

## Social Impact Report
The project includes a comprehensive Social Impact Report (available in the reports folder) that analyzes:

The societal impact of COVID-19 across different Indian states and demographics
How predictive modeling can support public health decision-making
Potential interventions based on prediction insights
Recommendations for resource allocation in high-risk areas
Long-term implications for health infrastructure planning

This report serves as a bridge between the technical modeling work and its practical applications for public health officials, policymakers, and community leaders.

## Notes

- The prediction system uses the most recent 46 data points to forecast future cases
- The model can forecast state-wise COVID-19 cases for upcoming days
- Testing positivity rate is estimated when direct testing data is not available

## Future Improvements

- Integration of more detailed vaccination data
- Inclusion of mobility data to better account for population movement
- Implementation of more sophisticated time series models
- Development of regional sub-models for more targeted predictions