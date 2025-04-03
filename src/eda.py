import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def perform_eda():
    # Load dataset
    df = pd.read_csv("data/processed_data.csv", low_memory=False)
    
    # Convert Date column to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs("output", exist_ok=True)
    
    # 1. COVID-19 Case Growth Rate Trend Over Time (Improved)
    if "Date" in df.columns and "Case Growth Rate" in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Aggregate by date to get national average
        daily_avg = df.groupby("Date")["Case Growth Rate"].mean().reset_index()
        
        # Plot with confidence interval
        sns.lineplot(x="Date", y="Case Growth Rate", data=daily_avg)
        
        # Add rolling average
        rolling_avg = daily_avg.set_index("Date")["Case Growth Rate"].rolling(window=7).mean()
        plt.plot(rolling_avg.index, rolling_avg.values, color='red', linewidth=2, label='7-day Rolling Average')
        
        plt.xticks(rotation=45)
        plt.title("Trend of COVID-19 Case Growth Rate Over Time (7-day Rolling Average)")
        plt.xlabel("Date")
        plt.ylabel("Case Growth Rate")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/case_growth_trend.png")
        plt.close()
    
    # 2. Correlation Heatmap of Key Features (Improved)
    # Select only meaningful features for correlation
    features = [
        "Case Growth Rate", "Testing Positivity Rate", "Youth_Population_Pct", 
        "Women_Tobacco_Use_Pct", "Vaccination Coverage", "Vaccination Effectiveness"
    ]
    
    # Filter to include only features that exist in the data
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[available_features].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=0.5,
            mask=mask
        )
        plt.title("Correlation Heatmap of Key COVID-19 Features")
        plt.tight_layout()
        plt.savefig("output/correlation_heatmap.png")
        plt.close()
    
    # 3. Distribution of Testing Positivity Rate (Improved)
    if "Testing Positivity Rate" in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Remove any remaining invalid values
        positivity_data = df["Testing Positivity Rate"].dropna()
        positivity_data = positivity_data[(positivity_data >= 0) & (positivity_data <= 1)]
        
        # Plot histogram with KDE
        sns.histplot(positivity_data, bins=30, kde=True)
        plt.title("Distribution of COVID-19 Testing Positivity Rate")
        plt.xlabel("Testing Positivity Rate")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output/testing_positivity_distribution.png")
        plt.close()
    
    # 4. NEW: Time-series of cases by state (top 5 states)
    if "Date" in df.columns and "State/UnionTerritory" in df.columns and "Confirmed" in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Get top 5 states by case count
        top_states = df.groupby("State/UnionTerritory")["Confirmed"].max().nlargest(5).index.tolist()
        
        # Filter and plot
        state_data = df[df["State/UnionTerritory"].isin(top_states)]
        sns.lineplot(x="Date", y="Confirmed", hue="State/UnionTerritory", data=state_data)
        
        plt.xticks(rotation=45)
        plt.title("COVID-19 Cases Over Time (Top 5 States)")
        plt.xlabel("Date")
        plt.ylabel("Confirmed Cases")
        plt.grid(True)
        plt.legend(title="State/UT")
        plt.tight_layout()
        plt.savefig("output/cases_by_state.png")
        plt.close()
    
    # 5. NEW: Relationship between testing positivity and case growth
    if "Testing Positivity Rate" in df.columns and "Case Growth Rate" in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Filter out extreme values
        scatter_data = df[
            (df["Testing Positivity Rate"] >= 0) & 
            (df["Testing Positivity Rate"] <= 1) &
            (df["Case Growth Rate"] >= 0) & 
            (df["Case Growth Rate"] <= 5)
        ]
        
        # Plot scatter with regression line
        sns.regplot(
            x="Testing Positivity Rate", 
            y="Case Growth Rate", 
            data=scatter_data,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"}
        )
        
        plt.title("Relationship Between Testing Positivity Rate and Case Growth Rate")
        plt.xlabel("Testing Positivity Rate")
        plt.ylabel("Case Growth Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output/positivity_vs_growth.png")
        plt.close()
    
    print("✅ EDA Visualizations Created & Saved!")

if __name__ == "__main__":
    perform_eda()