import pandas as pd
import numpy as np
import os

def load_and_clean_data():
    print("Loading and cleaning data...")
    
    # Load datasets
    df_covid = pd.read_csv("data/covid_19_india.csv", dtype={"Confirmed": float, "Deaths": float})
    df_health = pd.read_csv("data/datafile.csv")
    
    # Print column names to debug
    print("COVID data columns:", df_covid.columns.tolist())
    
    # Convert dates
    df_covid['Date'] = pd.to_datetime(df_covid['Date'])
    
    # Standardize column names
    if "State/UT" in df_health.columns:
        df_health.rename(columns={"State/UT": "State/UnionTerritory"}, inplace=True)
    
    # Merge health data
    df_merged = df_covid.merge(df_health, on="State/UnionTerritory", how="left")
    
    # Calculate daily new cases for each state
    df_merged = df_merged.sort_values(['State/UnionTerritory', 'Date'])
    df_merged['New_Cases'] = df_merged.groupby('State/UnionTerritory')['Confirmed'].diff().fillna(0)
    
    # Feature Engineering - correctly calculate metrics
    
    # Case Growth Rate: Current day's new cases / previous day's total cases
    df_merged['Previous_Day_Cases'] = df_merged.groupby('State/UnionTerritory')['Confirmed'].shift(1).fillna(1)
    df_merged["Case Growth Rate"] = df_merged["New_Cases"] / df_merged["Previous_Day_Cases"]
    
    # Testing Positivity Rate: Using actual test data if available, otherwise estimate
    if 'Tested' in df_merged.columns:
        print("Using 'Tested' column for positivity rate calculation")
        df_merged["Testing Positivity Rate"] = df_merged["New_Cases"] / df_merged["Tested"].clip(lower=1)
    else:
        # If test data and recovered data not available, use a safer estimate
        print("'Tested' column not found, using alternative calculation for positivity rate")
        df_merged["Testing Positivity Rate"] = df_merged["New_Cases"] / (df_merged["New_Cases"] * 10 + 1)
    
    # Cap positivity rate at 1.0 (100%)
    df_merged["Testing Positivity Rate"] = df_merged["Testing Positivity Rate"].clip(upper=1.0)
    
    # Handle vaccination data - if not available, create placeholder
    if 'Population below age 15 years (%)' in df_merged.columns:
        df_merged.rename(columns={
            "Population below age 15 years (%)": "Youth_Population_Pct"
        }, inplace=True)
    else:
        df_merged["Youth_Population_Pct"] = 0
        
    if 'Women age 15 years and above who use any kind of tobacco (%)' in df_merged.columns:
        df_merged.rename(columns={
            "Women age 15 years and above who use any kind of tobacco (%)": "Women_Tobacco_Use_Pct"
        }, inplace=True)
    else:
        df_merged["Women_Tobacco_Use_Pct"] = 0
    
    # Add placeholder for vaccination effectiveness (since we don't have actual data)
    df_merged["Vaccination Effectiveness"] = 0.5  # Default placeholder value
    
    # Add date-related features
    df_merged["Month"] = df_merged["Date"].dt.month
    df_merged["Week"] = df_merged["Date"].dt.isocalendar().week
    df_merged["Day_of_week"] = df_merged["Date"].dt.dayofweek
    first_date = df_merged["Date"].min()
    df_merged["Days_since_start"] = (df_merged["Date"] - first_date).dt.days
    
    # Ensure all numerical columns have the correct type
    numeric_columns = [
        "Case Growth Rate", "Testing Positivity Rate", 
        "Vaccination Effectiveness", "Youth_Population_Pct", 
        "Women_Tobacco_Use_Pct", "Month", "Week", 
        "Day_of_week", "Days_since_start"
    ]
    
    # Handle missing and extreme values for numeric columns
    for col in numeric_columns:
        if col in df_merged.columns:
            # Replace infinities with NaN first - UPDATED to avoid FutureWarning
            df_merged[col] = df_merged[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaNs with median for the column - UPDATED to avoid FutureWarning
            median_val = df_merged[col].median()
            if pd.isna(median_val):  # If median is also NaN, use 0
                df_merged[col] = df_merged[col].fillna(0)
            else:
                df_merged[col] = df_merged[col].fillna(median_val)
    
    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save processed data
    df_merged.to_csv("data/processed_data.csv", index=False)
    
    print("✅ Data Cleaning & Feature Engineering Completed!")
    print(f"Data saved to data/processed_data.csv with {len(df_merged)} rows and {len(df_merged.columns)} columns")

if __name__ == "__main__":
    load_and_clean_data()