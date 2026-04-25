import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar

def create_study_area_map():
    # Create figure with specific size
    fig = plt.figure(figsize=(14, 12))
    
    # Use PlateCarree projection for lat/lon
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set extent for your study area (adjust these coordinates)
    # Based on typical MTLCC study areas in Africa
    lon_min, lon_max = 10.5, 12.5  # Longitude range (11°E to 12°E)
    lat_min, lat_max = -5.0, -2.0   # Latitude range (5°S to 2°S)
    
    # Add margin (10% on each side)
    lon_margin = (lon_max - lon_min) * 0.1
    lat_margin = (lat_max - lat_min) * 0.1
    
    # Set map extent
    ax.set_extent([lon_min - lon_margin, lon_max + lon_margin,
                   lat_min - lat_margin, lat_max + lat_margin],
                  crs=ccrs.PlateCarree())
    
    # Add map features
    # Land (light gray)
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='black', linewidth=0.5)
    
    # Ocean (light blue)
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f0ff')
    
    # Lakes (blue)
    ax.add_feature(cfeature.LAKES, facecolor='#a0c0ff', edgecolor='blue', linewidth=0.5)
    
    # Rivers (blue lines)
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)
    
    # Borders (black lines)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    
    # Add gridlines with labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=1, color='black', alpha=0.5, linestyle='--')
    
    # Configure gridline labels
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    
    # Format labels with degrees and minutes
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Set label style
    gl.xlabel_style = {'size': 12, 'weight': 'bold'}
    gl.ylabel_style = {'size': 12, 'weight': 'bold'}
    
    # Define study area polygons (example coordinates)
    # Study Area 1 (green - replace with actual coordinates)
    study_area_1 = np.array([
        [11.0, -4.5],  # lon, lat
        [11.8, -4.5],
        [11.8, -3.8],
        [11.0, -3.8],
        [11.0, -4.5]
    ])
    
    # Study Area 2 (red - replace with actual coordinates)
    study_area_2 = np.array([
        [11.2, -3.5],
        [12.0, -3.5],
        [12.0, -2.8],
        [11.2, -2.8],
        [11.2, -3.5]
    ])
    
    # Study Area 3 (optional - yellow)
    study_area_3 = np.array([
        [10.8, -4.8],
        [11.5, -4.8],
        [11.5, -4.2],
        [10.8, -4.2],
        [10.8, -4.8]
    ])
    
    # Plot polygons
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    
    # Green polygon (with transparency)
    poly1 = Polygon(study_area_1, facecolor='green', edgecolor='darkgreen', 
                    alpha=0.5, linewidth=2, label='Study Area 1')
    ax.add_patch(poly1)
    
    # Red polygon (with transparency)
    poly2 = Polygon(study_area_2, facecolor='red', edgecolor='darkred', 
                    alpha=0.5, linewidth=2, label='Study Area 2')
    ax.add_patch(poly2)
    
    # Yellow polygon (optional)
    poly3 = Polygon(study_area_3, facecolor='yellow', edgecolor='orange', 
                    alpha=0.5, linewidth=2, label='Study Area 3')
    ax.add_patch(poly3)
    
    # Add north arrow
    x, y = 0.1, 0.9  # Position in axes coordinates
    ax.annotate('N', xy=(x, y), xycoords='axes fraction',
                fontsize=16, fontweight='bold',
                ha='center', va='center')
    ax.annotate('↑', xy=(x, y-0.02), xycoords='axes fraction',
                fontsize=20, ha='center', va='center')
    
    # Alternative: Draw north arrow with arrow patch
    north_arrow_x = lon_min - lon_margin + 0.1
    north_arrow_y = lat_max + lat_margin - 0.1
    ax.arrow(north_arrow_x, north_arrow_y, 0, 0.1, 
             head_width=0.05, head_length=0.05, 
             fc='black', ec='black', transform=ccrs.PlateCarree())
    ax.text(north_arrow_x, north_arrow_y + 0.12, 'N', 
            fontsize=12, fontweight='bold', ha='center', transform=ccrs.PlateCarree())
    
    # Add scale bar
    # Calculate scale bar length (e.g., 50 km)
    # Position at bottom right
    from cartopy.mpl.geoaxes import GeoAxesSubplot
    
    # Method 1: Using matplotlib_scalebar (if installed)
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        scalebar = ScaleBar(1, units='km', dimension='si-length',
                           location='lower right', scale_loc='bottom',
                           length_fraction=0.25, box_alpha=0.7)
        ax.add_artist(scalebar)
    except ImportError:
        # Manual scale bar if matplotlib_scalebar not installed
        # pip install matplotlib-scalebar to install
        print("Install matplotlib-scalebar for better scale bars: pip install matplotlib-scalebar")
        
        # Manual scale bar (simplified)
        scale_length_deg = 0.5  # ~55 km near equator
        scale_x = lon_max + lon_margin - 0.3
        scale_y = lat_min - lat_margin + 0.1
        
        ax.plot([scale_x, scale_x + scale_length_deg], [scale_y, scale_y], 
                'k-', linewidth=3, transform=ccrs.PlateCarree())
        ax.text(scale_x + scale_length_deg/2, scale_y - 0.05, '50 km',
                ha='center', transform=ccrs.PlateCarree(), fontsize=10)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.5, edgecolor='darkgreen', label='Study Area 1'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.5, edgecolor='darkred', label='Study Area 2'),
        plt.Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.5, edgecolor='orange', label='Study Area 3')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
              frameon=True, fontsize=10, title='Study areas of MTLCC dataset')
    
    # Add title
    plt.title('Study Areas of MTLCC Dataset', fontsize=14, pad=20, fontweight='bold')
    
    # Add inset map (optional) - for location context
    # Create inset axes
    inset_ax = fig.add_axes([0.15, 0.15, 0.2, 0.2], projection=ccrs.PlateCarree())
    inset_ax.set_extent([-20, 55, -35, 20], crs=ccrs.PlateCarree())
    inset_ax.add_feature(cfeature.LAND, facecolor='lightgray')
    inset_ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    inset_ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Mark study area location on inset
    study_region = plt.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                                  facecolor='red', alpha=0.5, edgecolor='red',
                                  transform=ccrs.PlateCarree())
    inset_ax.add_patch(study_region)
    inset_ax.gridlines(draw_labels=False)
    
    # Save figure
    plt.savefig('study_area_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Map created successfully!")

# Alternative: If you have shapefile with study areas
def create_map_from_shapefile(shapefile_path):
    """
    Create map from shapefile containing study area polygons
    """
    import geopandas as gpd
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Load study areas from shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Get bounds
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    lon_min, lat_min, lon_max, lat_max = bounds
    
    # Add margin
    lon_margin = (lon_max - lon_min) * 0.1
    lat_margin = (lat_max - lat_min) * 0.1
    
    # Set extent
    ax.set_extent([lon_min - lon_margin, lon_max + lon_margin,
                   lat_min - lat_margin, lat_max + lat_margin],
                  crs=ccrs.PlateCarree())
    
    # Add basemap features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.BORDERS, edgecolor='black')
    
    # Plot study areas from shapefile
    gdf.plot(ax=ax, facecolor='red', alpha=0.5, edgecolor='darkred', linewidth=2)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add scale bar and north arrow
    # ... (add similar elements as above)
    
    plt.savefig('study_area_map_shapefile.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Install required packages if not already installed:
    # pip install matplotlib cartopy geopandas matplotlib-scalebar
    
    create_study_area_map()
    
    # If you have a shapefile, use:
    # create_map_from_shapefile('path/to/your/study_areas.shp')