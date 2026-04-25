#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_satellite_data_explore.py
Objective: Explore and visualize real satellite time series data
Author: Amo
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

# Create screenshots folder if it doesn't exist
os.makedirs('screenshots', exist_ok=True)

print("=" * 60)
print("SATELLITE TIME SERIES DATA EXPLORATION")
print("=" * 60)

# -------------------------------------------------------------------
# 1. LOAD THE DATA
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. LOADING SATELLITE DATA")
print("=" * 60)

# Load the data (make sure your CSV file is in the same folder)
# If your file has a different name, change it here
data_file = "satellite_data.csv"  # Change this to your filename

try:
    df = pd.read_csv(data_file)
    print(f"✅ File loaded successfully: {data_file}")
except FileNotFoundError:
    print(f"❌ File not found: {data_file}")
    print("Please make sure your CSV file is in the current directory.")
    print("Looking for files:")
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            print(f"   - {file}")
    exit()

print(f"\nDataFrame shape: {df.shape}")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

# -------------------------------------------------------------------
# 2. EXAMINE COLUMNS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. EXAMINING COLUMNS")
print("=" * 60)

print("\nColumn names:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# -------------------------------------------------------------------
# 3. CHECK FOR MISSING VALUES
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. CHECKING FOR MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found!")

# -------------------------------------------------------------------
# 4. BASIC STATISTICS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. BASIC STATISTICS")
print("=" * 60)

# Select only numeric columns for statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nNumeric columns: {list(numeric_cols)}")

print("\nStatistical summary:")
print(df[numeric_cols].describe())

# -------------------------------------------------------------------
# 5. UNDERSTAND THE DATA STRUCTURE
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. UNDERSTANDING DATA STRUCTURE")
print("=" * 60)

# Check unique values in categorical columns
for col in df.columns:
    if col not in numeric_cols:
        unique_vals = df[col].nunique()
        print(f"\n{col}: {unique_vals} unique values")
        if unique_vals < 10:  # Show all if small number
            print(f"   Values: {df[col].unique()}")

# Check for date column
date_cols = [col for col in df.columns if 'date' in col.lower()]
if date_cols:
    print(f"\nPossible date column: {date_cols[0]}")
    # Try to convert to datetime
    try:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        print(f"✅ Converted {date_cols[0]} to datetime")
        print(f"Date range: {df[date_cols[0]].min()} to {df[date_cols[0]].max()}")
    except:
        print(f"⚠️ Could not convert {date_cols[0]} to datetime")

# -------------------------------------------------------------------
# 6. IDENTIFY TIME SERIES VARIABLES
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. IDENTIFYING TIME SERIES VARIABLES")
print("=" * 60)

# Look for vegetation indices or other time-varying variables
possible_ts_vars = ['vim', 'vim_avg', 'viq', 'ndvi', 'evi', 'lai', 'fpar']
found_ts_vars = []

for var in possible_ts_vars:
    if var in df.columns:
        found_ts_vars.append(var)
        print(f"✅ Found time series variable: {var}")

if not found_ts_vars:
    print("No common vegetation indices found. Using all numeric columns:")
    found_ts_vars = list(numeric_cols)
    for var in found_ts_vars:
        print(f"   - {var}")

# -------------------------------------------------------------------
# 7. VISUALIZE TIME SERIES FOR DIFFERENT REGIONS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. VISUALIZING TIME SERIES")
print("=" * 60)

# Identify region/ID columns
id_cols = [col for col in df.columns if 'id' in col.lower() or 'code' in col.lower() or 'adm' in col.lower()]
print(f"Possible region identifier columns: {id_cols}")

if id_cols and len(found_ts_vars) > 0:
    # Choose the first ID column
    region_col = id_cols[0]
    
    # Get unique regions
    unique_regions = df[region_col].unique()
    print(f"\nFound {len(unique_regions)} unique regions")
    
    # Select a few regions to plot (first 3)
    n_regions_to_plot = min(3, len(unique_regions))
    regions_to_plot = unique_regions[:n_regions_to_plot]
    
    # Create figure
    fig, axes = plt.subplots(n_regions_to_plot, len(found_ts_vars), 
                              figsize=(5*len(found_ts_vars), 4*n_regions_to_plot))
    
    # Handle case when only one subplot
    if n_regions_to_plot == 1 and len(found_ts_vars) == 1:
        axes = np.array([[axes]])
    elif n_regions_to_plot == 1:
        axes = axes.reshape(1, -1)
    elif len(found_ts_vars) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, region in enumerate(regions_to_plot):
        region_data = df[df[region_col] == region].copy()
        
        # Sort by date if available
        if date_cols:
            region_data = region_data.sort_values(date_cols[0])
            x_data = region_data[date_cols[0]]
        else:
            x_data = range(len(region_data))
        
        for j, var in enumerate(found_ts_vars):
            ax = axes[i, j]
            ax.plot(x_data, region_data[var], marker='o', markersize=3, linewidth=1)
            ax.set_title(f'Region {region}: {var}')
            ax.set_xlabel('Date' if date_cols else 'Time step')
            ax.set_ylabel(var)
            ax.grid(True, alpha=0.3)
            
            # Rotate dates if needed
            if date_cols and len(region_data) > 10:
                ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('screenshots/05_regions_time_series.png', dpi=150)
    plt.show()
    print("✓ Saved: screenshots/05_regions_time_series.png")
    
else:
    print("⚠️ Could not identify region columns or time series variables")

# -------------------------------------------------------------------
# 8. VISUALIZE SEASONAL PATTERNS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. ANALYZING SEASONAL PATTERNS")
print("=" * 60)

if date_cols and found_ts_vars:
    # Extract month and year
    df['month'] = df[date_cols[0]].dt.month
    df['year'] = df[date_cols[0]].dt.year
    
    # Plot monthly patterns for each variable
    fig, axes = plt.subplots(1, len(found_ts_vars), figsize=(6*len(found_ts_vars), 5))
    if len(found_ts_vars) == 1:
        axes = [axes]
    
    for i, var in enumerate(found_ts_vars):
        # Group by month and calculate mean
        monthly_mean = df.groupby('month')[var].mean()
        monthly_std = df.groupby('month')[var].std()
        
        ax = axes[i]
        ax.bar(monthly_mean.index, monthly_mean.values, yerr=monthly_std.values, 
               capsize=5, alpha=0.7, color='green', edgecolor='black')
        ax.set_title(f'Monthly Pattern: {var}')
        ax.set_xlabel('Month')
        ax.set_ylabel(var)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('screenshots/05_seasonal_patterns.png', dpi=150)
    plt.show()
    print("✓ Saved: screenshots/05_seasonal_patterns.png")

# -------------------------------------------------------------------
# 9. CORRELATION ANALYSIS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("9. CORRELATION ANALYSIS")
print("=" * 60)

if len(numeric_cols) > 1:
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    print("\nCorrelation matrix:")
    print(corr_matrix)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Satellite Variables')
    plt.tight_layout()
    plt.savefig('screenshots/05_correlation_matrix.png', dpi=150)
    plt.show()
    print("✓ Saved: screenshots/05_correlation_matrix.png")
    
    # Highlight strong correlations
    print("\nStrong correlations (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                print(f"  {corr_matrix.columns[i]} & {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

# -------------------------------------------------------------------
# 10. PREPARE DATA FOR LSTM (PREVIEW)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("10. PREPARING FOR LSTM (PREVIEW)")
print("=" * 60)

if found_ts_vars and date_cols and id_cols:
    # Choose first region and first variable for example
    first_region = unique_regions[0]
    first_var = found_ts_vars[0]
    
    print(f"\nPreparing data for region {first_region}, variable {first_var}")
    
    # Extract time series for this region
    region_data = df[df[region_col] == first_region].copy()
    region_data = region_data.sort_values(date_cols[0])
    
    ts_values = region_data[first_var].values
    dates = region_data[date_cols[0]].values
    
    print(f"Time series length: {len(ts_values)} points")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Value range: [{ts_values.min():.4f}, {ts_values.max():.4f}]")
    
    # Check for missing values
    missing = np.isnan(ts_values).sum()
    if missing > 0:
        print(f"⚠️ Found {missing} missing values")
    
    # Normalize data (preview)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    ts_normalized = scaler.fit_transform(ts_values.reshape(-1, 1)).flatten()
    print(f"Normalized range: [{ts_normalized.min():.4f}, {ts_normalized.max():.4f}]")
    
    # Plot original vs normalized
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(dates, ts_values, marker='o', markersize=3, linewidth=1)
    ax1.set_title(f'Original {first_var} - Region {first_region}')
    ax1.set_ylabel(first_var)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(dates, ts_normalized, marker='o', markersize=3, linewidth=1, color='orange')
    ax2.set_title(f'Normalized {first_var} (range [-1, 1])')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized value')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('screenshots/05_data_preparation_preview.png', dpi=150)
    plt.show()
    print("✓ Saved: screenshots/05_data_preparation_preview.png")

# -------------------------------------------------------------------
# 11. SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("11. SUMMARY - DATA EXPLORATION COMPLETE")
print("=" * 60)

print(f"""
📊 Dataset Summary:
   - Total samples: {len(df)}
   - Number of regions: {len(unique_regions) if id_cols else 'Unknown'}
   - Time series variables: {found_ts_vars}
   - Date range: {df[date_cols[0]].min()} to {df[date_cols[0]].max() if date_cols else 'Unknown'}

📈 Visualizations saved:
   - screenshots/05_regions_time_series.png
   - screenshots/05_seasonal_patterns.png
   - screenshots/05_correlation_matrix.png
   - screenshots/05_data_preparation_preview.png

🔜 Next step: 06_lstm_satellite.py
   - Apply LSTM to real satellite data
   - Predict vegetation indices
   - Evaluate on test regions
""")

print("\n" + "=" * 60)
print("DATA EXPLORATION COMPLETE! 🛰️")
print("=" * 60)