import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.patches as mpatches

# Define Bururi District approximate boundaries (Burundi)
# Coordinates: Bururi is around 3.9°S to 4.1°S, 29.5°E to 29.8°E
# For visualization, we'll show a larger area of Burundi with Bururi highlighted

# Create the figure and axis with PlateCarree projection
fig = plt.figure(figsize=(12, 10), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set the map extent for Burundi region
# Longitude: 28.5°E to 31.0°E, Latitude: 4.5°S to 2.0°S
ax.set_extent([28.5, 31.0, -4.5, -2.0], crs=ccrs.PlateCarree())

# Add map features
# Ocean, land, lakes, rivers
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.7, edgecolor='blue')
ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

# Add administrative boundaries (country borders)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1.5, linestyle='-')
ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.8, linestyle='--')

# For Bururi District, we'll create a custom polygon 
# (approximate boundaries - you can replace with actual shapefile)
bururi_lons = [29.5, 29.7, 29.8, 29.6, 29.5]
bururi_lats = [-3.8, -3.9, -4.0, -4.1, -3.8]

# Plot Bururi District as highlighted area
ax.fill(bururi_lons, bururi_lats, color='red', alpha=0.5, 
        transform=ccrs.PlateCarree(), label='Bururi District')

# Add boundary line for Bururi District
ax.plot(bururi_lons + [bururi_lons[0]], bururi_lats + [bururi_lats[0]], 
        'r-', linewidth=2, transform=ccrs.PlateCarree())

# Mark major cities
cities = {
    'Gitega': [29.93, -3.43],
    'Bujumbura': [29.36, -3.38],
    'Ruyigi': [30.25, -3.48],
    'Bururi': [29.62, -3.95],
    'Makamba': [29.80, -4.13],
    'Rutana': [30.00, -3.93]
}

for city, (lon, lat) in cities.items():
    ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree())
    ax.text(lon + 0.03, lat + 0.03, city, fontsize=9, 
            transform=ccrs.PlateCarree(), fontweight='bold')

# Add gridlines with labels
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                  linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}

# Add scale bar
scale_bar_length_km = 50
scale_bar_length_deg = scale_bar_length_km / 111  # 1 degree ≈ 111 km
x_start = 28.8
y_start = -4.3

ax.plot([x_start, x_start + scale_bar_length_deg], [y_start, y_start], 
        'k-', linewidth=3, transform=ccrs.PlateCarree())
ax.plot([x_start, x_start], [y_start - 0.03, y_start + 0.03], 
        'k-', linewidth=2, transform=ccrs.PlateCarree())
ax.plot([x_start + scale_bar_length_deg, x_start + scale_bar_length_deg], 
        [y_start - 0.03, y_start + 0.03], 'k-', linewidth=2, 
        transform=ccrs.PlateCarree())
ax.text(x_start + scale_bar_length_deg/2, y_start - 0.07, 
        f'{scale_bar_length_km} km', fontsize=9, ha='center', 
        transform=ccrs.PlateCarree())

# Add north arrow
x_north = 28.7
y_north = -2.2
ax.annotate('N', xy=(x_north, y_north), xytext=(x_north, y_north - 0.15),
            arrowprops=dict(facecolor='black', width=3, headwidth=8),
            fontsize=14, fontweight='bold', ha='center',
            transform=ccrs.PlateCarree())

# Add title
plt.title('Study Area: Bururi District, Burundi', fontsize=16, 
          fontweight='bold', pad=20)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='red', alpha=0.5, edgecolor='red', 
                   label='Bururi District (Study Area)'),
    mpatches.Patch(facecolor='lightblue', alpha=0.5, edgecolor='blue', 
                   label='Lake/Water Body'),
    mpatches.Patch(facecolor='lightgray', alpha=0.3, edgecolor='black', 
                   label='Land Area')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
          framealpha=0.9)

# Add coordinate information text box
text_box = f"Geographic Coordinates:\nLongitude: 28.5°E - 31.0°E\nLatitude: 4.5°S - 2.0°S"
ax.text(0.02, 0.98, text_box, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save the figure
plt.tight_layout()
plt.savefig('study_area_map3.png', dpi=300, bbox_inches='tight')
plt.savefig('study_area_map3.pdf', bbox_inches='tight')  # Also save as PDF for better quality
plt.show()

print("Map saved as 'study_area_map3.png' and 'study_area_map3.pdf'")