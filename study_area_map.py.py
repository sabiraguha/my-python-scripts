import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

# Use built-in world map (no shapefile needed)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
burundi = world[world['name'] == 'Burundi']

fig, ax = plt.subplots(figsize=(8, 6))
burundi.boundary.plot(ax=ax, edgecolor='red', linewidth=2)
ax.set_aspect('equal')

# Add scale bar
scalebar = ScaleBar(dx=1, units="km", location='lower right',
                    scale_loc='bottom', frameon=True)
ax.add_artist(scalebar)

# North arrow
ax.annotate('N', xy=(0.92, 0.92), xycoords='axes fraction',
            fontsize=12, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.title("Study area of Bururi District, Burundi")
plt.savefig('burundi_map.png', dpi=300, bbox_inches='tight')
plt.show()