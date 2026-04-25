import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

# ==========================================
# 1. Load your shapefile (adjust the path!)
# ==========================================
# If you have a shapefile, use:
study_area = gpd.read_file("bururi_district.shp")

# ==========================================
# 2. If you DON'T have a shapefile yet,
#    you can download a simple country outline
#    as a placeholder (uncomment the lines below):
# ==========================================
# import requests
# url = "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/admin/gb/wales.json"
# # This is just an example – you need Burundi data.
# # For a proper Burundi outline, download from GADM (https://gadm.org).
# # Then replace the line above.

# Create the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the boundary
study_area.boundary.plot(ax=ax, edgecolor='red', linewidth=2)

# Set aspect ratio (keep shape proportions)
ax.set_aspect('equal')

# Add a scale bar
scalebar = ScaleBar(dx=1, units="km", location='lower right',
                    scale_loc='bottom', frameon=True)
ax.add_artist(scalebar)

# Add a north arrow
ax.annotate('N', xy=(0.92, 0.92), xycoords='axes fraction',
            fontsize=12, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Title
plt.title("Study area of Bururi District, Burundi")

# Save the figure
plt.savefig('study_area_map.png', dpi=300, bbox_inches='tight')
plt.show()