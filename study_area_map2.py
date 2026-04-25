import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.patches as mpatches

# ============================================
# STEP 1: Load your existing GeoJSON file
# ============================================
print("Loading GeoJSON file...")
gdf = gpd.read_file("C:/Users/ponhu/Desktop/semaine1_pytorch/PASTIS (1)/gadm41_BDI_3.json")

# Print info about the data
print(f"Loaded {len(gdf)} features")
print(f"Columns available: {gdf.columns.tolist()}")
print(f"Bounding box: {gdf.total_bounds}")

# ============================================
# STEP 2: Extract Bururi district from the GeoJSON
# ============================================
# Check if 'NAME_1' column exists (it does in your GeoJSON)
if 'NAME_1' in gdf.columns:
    # Filter to get only Bururi district
    bururi_district = gdf[gdf['NAME_1'] == 'Bururi']
    
    if len(bururi_district) > 0:
        print(f"Found Bururi district with {len(bururi_district)} features")
    else:
        print("Bururi not found. Available regions:", gdf['NAME_1'].unique())
        # If Bururi not found, use all data
        print("Using all data instead...")
        bururi_district = gdf
else:
    print("'NAME_1' column not found. Using all data.")
    bururi_district = gdf

# ============================================
# STEP 3: Define your 3 study areas (based on your map)
# ============================================
# Get the bounds of Bururi to set reasonable coordinates
bounds = bururi_district.total_bounds
print(f"Bururi bounds: {bounds}")

# Define study areas (adjust these coordinates based on your actual map)
# These are approximate - you should adjust them based on your study_area_map.png
area1 = box(bounds[0] + 0.1, bounds[1] + 0.1, bounds[0] + 0.3, bounds[1] + 0.3)  # Green
area2 = box(bounds[0] + 0.3, bounds[1] + 0.2, bounds[0] + 0.5, bounds[1] + 0.4)  # Red
area3 = box(bounds[0] + 0.5, bounds[1] + 0.3, bounds[0] + 0.7, bounds[1] + 0.5)  # Yellow

# Create GeoDataFrame for study areas
study_areas = gpd.GeoDataFrame({
    'name': ['Study Area 1 (Green)', 'Study Area 2 (Red)', 'Study Area 3 (Yellow)'],
    'color': ['green', 'red', 'gold'],
    'geometry': [area1, area2, area3]
}, crs='EPSG:4326')

# ============================================
# STEP 4: Create the map
# ============================================
fig, ax = plt.subplots(figsize=(14, 12))

# Plot Burundi/Bururi boundaries
bururi_district.boundary.plot(ax=ax, color='black', linewidth=0.8, alpha=0.7)

# Plot the study areas
study_areas.plot(ax=ax, color=study_areas['color'], alpha=0.4, edgecolor='black', linewidth=2)

# Add labels for study areas
for idx, row in study_areas.iterrows():
    ax.annotate(row['name'], 
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                fontsize=10, ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

# Add title and labels
ax.set_title('Study Areas of the MTLCC Dataset', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°S)', fontsize=12)

# Set axis limits to focus on the study area
ax.set_xlim(bounds[0] - 0.2, bounds[2] + 0.2)
ax.set_ylim(bounds[1] - 0.2, bounds[3] + 0.2)

# Add grid
ax.grid(True, linestyle='--', alpha=0.5)

# Add legend
legend_elements = [
    mpatches.Patch(color='green', alpha=0.4, label='Study Area 1 (Green)'),
    mpatches.Patch(color='red', alpha=0.4, label='Study Area 2 (Red)'),
    mpatches.Patch(color='gold', alpha=0.4, label='Study Area 3 (Yellow)'),
    mpatches.Patch(color='black', alpha=0.7, label='Administrative Boundary')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()

# ============================================
# STEP 5: Save the map (as study_area_map2.png)
# ============================================
plt.savefig('study_area_map2.png', dpi=300, bbox_inches='tight')
print("Map saved as 'study_area_map2.png'")

# Also save as PDF for better quality (optional)
plt.savefig('study_area_map2.pdf', bbox_inches='tight')
print("Also saved as 'study_area_map2.pdf'")

plt.show()
print("Done!")