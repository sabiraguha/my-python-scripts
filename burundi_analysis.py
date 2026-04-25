"""
Research Analysis Pipeline for Burundi GADM Data
Author: [Your Name]
Date: [Current Date]
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_clean_data(filepath):
    """Load and clean GADM data"""
    gdf = gpd.read_file(filepath)
    gdf = gdf[gdf['TYPE_3'] != 'Waterbody'].copy()
    return gdf

def calculate_areas(gdf):
    """Calculate areas in square kilometers"""
    gdf_proj = gdf.to_crs('EPSG:32736')
    gdf['area_km2'] = gdf_proj.geometry.area / 1_000_000
    return gdf

def create_study_area(gdf, province_name=None, commune_name=None):
    """Extract study area based on province or commune"""
    if province_name:
        return gdf[gdf['NAME_1'] == province_name]
    elif commune_name:
        return gdf[gdf['NAME_2'] == commune_name]
    else:
        return gdf

def main():
    # Load data
    print("Loading data...")
    burundi = load_and_clean_data('gadm41_BDI_3.json')
    
    # Calculate areas
    print("Calculating areas...")
    burundi = calculate_areas(burundi)
    
    # Generate statistics
    print("\n=== STUDY AREA STATISTICS ===")
    print(f"Total land area: {burundi['area_km2'].sum():.0f} km²")
    print(f"Number of collines: {len(burundi)}")
    print(f"Number of communes: {burundi['NAME_2'].nunique()}")
    print(f"Number of provinces: {burundi['NAME_1'].nunique()}")
    
    # Save cleaned data
    burundi.to_file('burundi_cleaned.gpkg', driver='GPKG')
    print("\nCleaned data saved to 'burundi_cleaned.gpkg'")
    
    # Generate summary table
    summary = burundi.groupby('NAME_1').agg({
        'NAME_2': 'nunique',
        'NAME_3': 'nunique',
        'area_km2': 'sum'
    }).round(2)
    summary.columns = ['Communes', 'Collines', 'Area_km2']
    summary.to_csv('province_summary.csv')
    print("Summary table saved to 'province_summary.csv'")

if __name__ == "__main__":
    main()