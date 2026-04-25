import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def create_burundi_map_debug():
    # Try to get Burundi boundaries
    # If you have a shapefile or other data source, check those values
    
    # Option 1: Hard-coded boundaries (most reliable)
    lon_min, lon_max = 28.5, 30.5
    lat_min, lat_max = -4.5, -2.0
    
    # Option 2: If loading from somewhere, check values
    # Example with error checking:
    try:
        # Replace this with your actual data loading
        # lon_min = your_data['lon_min']
        # lon_max = your_data['lon_max']
        # lat_min = your_data['lat_min']
        # lat_max = your_data['lat_max']
        
        # Validate loaded values
        if not all(isinstance(x, (int, float)) for x in [lon_min, lon_max, lat_min, lat_max]):
            print("Warning: Non-numeric values found, using defaults")
            lon_min, lon_max = 28.5, 30.5
            lat_min, lat_max = -4.5, -2.0
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using default boundaries")
        lon_min, lon_max = 28.5, 30.5
        lat_min, lat_max = -4.5, -2.0
    
    # Add small margin to avoid boundaries exactly at data edges
    lon_margin = 0.5  # Fixed margin instead of percentage
    lat_margin = 0.5
    
    # Create figure with specific projection
    fig = plt.figure(figsize=(12, 10))
    
    # Use PlateCarree for simple lat/lon projection
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    
    # Set extent with margins
    extent = [lon_min - lon_margin, lon_max + lon_margin, 
              lat_min - lat_margin, lat_max + lat_margin]
    
    print(f"Final extent: {extent}")
    
    # Check for NaN/Inf
    if not all(np.isfinite(extent)):
        print("ERROR: Extent contains NaN or Inf values")
        print(f"NaN check: {[np.isnan(x) for x in extent]}")
        print(f"Inf check: {[np.isinf(x) for x in extent]}")
        return
    
    try:
        ax.set_extent(extent, crs=projection)
        
        # Add features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1.5)
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='blue')
        
        # Add gridlines
        gl = ax.gridlines(crs=projection, draw_labels=True, 
                         linewidth=1, color='gray', alpha=0.7, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Add title
        ax.set_title('Burundi Map', fontsize=14, pad=20)
        
        # Save and show
        plt.savefig('burundi_map.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Map created successfully!")
        
    except ValueError as e:
        print(f"Error in set_extent: {e}")
        print(f"Extent values: {extent}")
        
        # Try with a default extent
        print("Attempting with default extent...")
        default_extent = [28, 31, -5, -1]
        ax.set_extent(default_extent, crs=projection)
        
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.BORDERS, edgecolor='black')
        ax.set_title('Burundi Map (with default extent)')
        
        plt.savefig('burundi_map_default.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    create_burundi_map_debug()