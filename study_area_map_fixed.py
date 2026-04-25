import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches

# ========== CHANGED: Smaller figure size and lower DPI ==========
# Original: figsize=(12, 10), dpi=300 -> Large file (~2-5 MB)
# New: figsize=(8, 6), dpi=150 -> Smaller file (~200-500 KB)
fig = plt.figure(figsize=(8, 6), dpi=150)  # REDUCED for smaller file size
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set map extent for Burundi (zoomed correctly)
# Longitude: 28.5°E to 31.0°E, Latitude: 4.5°S to 2.0°S
ax.set_extent([28.5, 31.0, -4.5, -2.0], crs=ccrs.PlateCarree())

# ========== USE SIMPLER FEATURES (faster, more reliable) ==========
# Add ocean and land with simpler settings
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5, edgecolor='none')
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.4, edgecolor='none')

# Add lakes (simpler - use 'lakes' feature)
ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.7, edgecolor='blue', linewidth=0.5)

# Add country borders (simpler resolution)
ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1.5, linestyle='-')

# Add rivers (optional, may be slow)
try:
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5, alpha=0.7)
except:
    print("Rivers feature not available - continuing without rivers")

# ========== Bururi District (approximate polygon) ==========
# More accurate coordinates for Bururi District
bururi_lons = [29.45, 29.70, 29.85, 29.75, 29.50, 29.45]
bururi_lats = [-3.80, -3.85, -3.95, -4.10, -4.05, -3.80]

# Fill Bururi District
ax.fill(bururi_lons, bururi_lats, color='red', alpha=0.5, 
        transform=ccrs.PlateCarree())

# Add boundary line
ax.plot(bururi_lons + [bururi_lons[0]], bururi_lats + [bururi_lats[0]], 
        'r-', linewidth=2.5, transform=ccrs.PlateCarree())

# ========== Cities ==========
cities = {
    'Bujumbura': (29.36, -3.38),
    'Gitega': (29.93, -3.43),
    'Bururi': (29.62, -3.95),
    'Rutana': (30.00, -3.93),
    'Makamba': (29.80, -4.13),
    'Ruyigi': (30.25, -3.48)
}

# ========== CHANGED: Slightly smaller city markers and text ==========
for city, (lon, lat) in cities.items():
    ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree())  # was 6
    ax.text(lon + 0.04, lat + 0.02, city, fontsize=8,  # was 10
            transform=ccrs.PlateCarree(), fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

# ========== Gridlines ==========
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                  linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8, 'color': 'black'}  # was 10
gl.ylabel_style = {'size': 8, 'color': 'black'}  # was 10

# ========== Scale Bar (50 km) ==========
# 1 degree longitude at equator ≈ 111 km. At 3°S, approx 110.9 km
scale_km = 50
scale_deg = scale_km / 111  # approximate

x_start, y_start = 28.65, -4.35
x_end = x_start + scale_deg

ax.plot([x_start, x_end], [y_start, y_start], 'k-', linewidth=2, 
        transform=ccrs.PlateCarree())  # was 3
ax.plot([x_start, x_start], [y_start - 0.03, y_start + 0.03], 'k-', linewidth=1.5,
        transform=ccrs.PlateCarree())  # was 2
ax.plot([x_end, x_end], [y_start - 0.03, y_start + 0.03], 'k-', linewidth=1.5,
        transform=ccrs.PlateCarree())  # was 2
ax.text(x_start + scale_deg/2, y_start - 0.09, f'{scale_km} km', 
        fontsize=7, ha='center', transform=ccrs.PlateCarree(),  # was 9
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# ========== North Arrow ==========
x_north, y_north = 28.65, -2.15
ax.annotate('N', xy=(x_north, y_north), xytext=(x_north, y_north - 0.15),
            arrowprops=dict(facecolor='black', width=2, headwidth=6, headlength=6),  # was 3,8,8
            fontsize=11, fontweight='bold', ha='center',  # was 14
            transform=ccrs.PlateCarree())

# ========== CHANGED: Smaller title ==========
plt.title('Study Area: Bururi District, Burundi', fontsize=12,  # was 16
          fontweight='bold', pad=15)  # was 20

# ========== Legend ==========
legend_elements = [
    mpatches.Patch(facecolor='red', alpha=0.5, edgecolor='red', linewidth=1.5,
                   label='Bururi District (Study Area)'),
    mpatches.Patch(facecolor='lightblue', alpha=0.5, edgecolor='blue',
                   label='Lake / Water Body'),
    mpatches.Patch(facecolor='lightgray', alpha=0.4, edgecolor='black',
                   label='Land Area')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7,  # was 9
          framealpha=0.9, edgecolor='black')

# ========== Information Box ==========
info_text = "Geographic Coordinates:\nLongitude: 28.5°E - 31.0°E\nLatitude: 4.5°S - 2.0°S"
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=6,  # was 8
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

# ========== Save with lower DPI for even smaller file ==========
plt.tight_layout()
plt.savefig('study_area_map3.png', dpi=150, bbox_inches='tight')  # CHANGED: 150 instead of 300
plt.savefig('study_area_map3.pdf', bbox_inches='tight')
plt.show()

print("\n✅ SUCCESS! Map saved as 'study_area_map3.png' and 'study_area_map3.pdf'")
print("   File size is now SMALLER (lower DPI = 150, smaller figure size)")