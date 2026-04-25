#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor basics with GeoJSON data.
Loads metadata.geojson, extracts numeric features, and applies PyTorch tensor operations.
Additionally, exports all properties to CSV for further analysis.
"""

import json
import csv
import torch
import numpy as np
from datetime import datetime

# =============================================================================
# 1. Load and inspect the GeoJSON file
# =============================================================================
print("=" * 60)
print("LOADING GEOJSON DATA")
print("=" * 60)

try:
    with open('metadata.geojson', 'r') as f:
        geojson = json.load(f)
    print(f"✅ Successfully loaded {len(geojson['features'])} features.")
except FileNotFoundError:
    print("❌ Error: metadata.geojson not found in the current directory.")
    exit(1)

# =============================================================================
# 2. Extract all properties (including dates) and optionally geometry to CSV
# =============================================================================
print("\nExtracting all properties for CSV export...")

# Prepare a list to hold each feature's properties as a flat dictionary
csv_rows = []

for feat in geojson['features']:
    props = feat['properties'].copy()
    
    # Add an ID if not already present (from feature id or property id)
    props.setdefault('feature_id', feat.get('id', ''))
    
    # Flatten dates-S2: convert to a JSON string (or you can expand into columns)
    if 'dates-S2' in props and isinstance(props['dates-S2'], dict):
        props['dates-S2_json'] = json.dumps(props['dates-S2'])
        # Optionally remove the original dict to avoid nested structure
        del props['dates-S2']
    
    # Add centroid coordinates (approximate from first polygon)
    geom = feat['geometry']
    if geom['type'] == 'MultiPolygon':
        coords = geom['coordinates'][0][0]
    elif geom['type'] == 'Polygon':
        coords = geom['coordinates'][0]
    else:
        coords = None
    
    if coords:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        props['centroid_x'] = sum(xs)/len(xs)
        props['centroid_y'] = sum(ys)/len(ys)
    
    # Add geometry as WKT (optional, can be large)
    # from shapely.geometry import shape
    # props['geometry_wkt'] = shape(geom).wkt
    
    csv_rows.append(props)

# Write CSV
csv_file = 'metadata_extracted.csv'
fieldnames = set()
for row in csv_rows:
    fieldnames.update(row.keys())
fieldnames = sorted(fieldnames)  # sort for consistency

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"✅ Saved {len(csv_rows)} rows to '{csv_file}'")
print(f"Columns: {', '.join(fieldnames)}")

# =============================================================================
# 3. Extract numeric data for tensor operations (as in original template)
# =============================================================================
print("\nExtracting numeric features for tensors...")

n_parcels = []
parcel_covers = []
centroids = []  # (x, y) coordinates of polygon centroids

for feat in geojson['features']:
    props = feat['properties']
    try:
        n_parcels.append(props['N_Parcel'])
        parcel_covers.append(props['Parcel_Cover'])
    except KeyError:
        print(f"Warning: missing N_Parcel or Parcel_Cover in feature {feat.get('id', 'unknown')}")
        continue

    geom = feat['geometry']
    if geom['type'] == 'MultiPolygon':
        coords = geom['coordinates'][0][0]
    elif geom['type'] == 'Polygon':
        coords = geom['coordinates'][0]
    else:
        print(f"Warning: unsupported geometry type {geom['type']} in feature {feat.get('id', 'unknown')}")
        continue

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    centroids.append((sum(xs)/len(xs), sum(ys)/len(ys)))

# Convert to numpy arrays
n_parcels = np.array(n_parcels, dtype=np.int32)
parcel_covers = np.array(parcel_covers, dtype=np.float32)
centroids = np.array(centroids, dtype=np.float32)

print(f"Extracted {len(n_parcels)} valid records for tensors.")
print(f"Sample N_Parcel values: {n_parcels[:5]}")
print(f"Sample Parcel_Cover values: {parcel_covers[:5]:.3f}")
print(f"Sample centroids: {centroids[:2]}")

# =============================================================================
# 4. Create PyTorch tensors
# =============================================================================
print("\n" + "=" * 60)
print("CREATING PYTORCH TENSORS")
print("=" * 60)

t_n_parcels = torch.from_numpy(n_parcels)
t_covers = torch.from_numpy(parcel_covers)
t_centroids = torch.from_numpy(centroids)

print(f"t_n_parcels shape: {t_n_parcels.shape}, dtype: {t_n_parcels.dtype}")
print(f"t_covers shape: {t_covers.shape}, dtype: {t_covers.dtype}")
print(f"t_centroids shape: {t_centroids.shape}, dtype: {t_centroids.dtype}")

# =============================================================================
# 5. Tensor properties (as in the tutorial)
# =============================================================================
print("\n" + "=" * 60)
print("TENSOR PROPERTIES")
print("=" * 60)

print(f"t_n_parcels: shape={t_n_parcels.shape}, ndim={t_n_parcels.ndim}, numel={t_n_parcels.numel()}")
print(f"t_covers: shape={t_covers.shape}, ndim={t_covers.ndim}, numel={t_covers.numel()}")

# =============================================================================
# 6. Indexing and slicing
# =============================================================================
print("\n" + "=" * 60)
print("INDEXING AND SLICING")
print("=" * 60)

print("First 5 Parcel_Cover values:", t_covers[:5])
print("N_Parcel values where Parcel_Cover > 0.9:", t_n_parcels[t_covers > 0.9])

# Boolean mask
mask = t_covers > 0.8
print(f"Number of parcels with cover > 0.8: {mask.sum().item()}")

# =============================================================================
# 7. Basic mathematical operations
# =============================================================================
print("\n" + "=" * 60)
print("BASIC MATHEMATICAL OPERATIONS")
print("=" * 60)

print(f"Sum of all N_Parcel: {t_n_parcels.sum().item()}")
print(f"Mean Parcel_Cover: {t_covers.mean().item():.4f}")
print(f"Standard deviation Parcel_Cover: {t_covers.std().item():.4f}")

# Elementwise operations
t_covers_shifted = t_covers + 0.1
print(f"Parcel_Cover + 0.1 (first 5): {t_covers_shifted[:5]:.3f}")

# Product of N_Parcel and Parcel_Cover
product = t_n_parcels * t_covers
print(f"N_Parcel * Parcel_Cover (first 5): {product[:5]:.2f}")

# 2D example: create a matrix from centroids (first 100 centroids)
centroids_100 = t_centroids[:100]
print(f"Centroids matrix (100 x 2): shape {centroids_100.shape}")

# Compute pairwise distances (illustration)
diffs = centroids_100.unsqueeze(1) - centroids_100.unsqueeze(0)  # (100,100,2)
distances = torch.sqrt((diffs**2).sum(dim=2))
print(f"Pairwise distance matrix shape: {distances.shape}")
print(f"Min distance (non-zero): {distances[distances>0].min().item():.2f}")

# =============================================================================
# 8. Reshaping and manipulation
# =============================================================================
print("\n" + "=" * 60)
print("RESHAPING AND MANIPULATION")
print("=" * 60)

# Reshape first 400 covers into 40 groups of 10
if len(t_covers) >= 400:
    covers_400 = t_covers[:400]
    covers_40x10 = covers_400.reshape(40, 10)
    print(f"Reshaped (40x10): {covers_40x10.shape}")
    print("Mean per group (dim=1):", covers_40x10.mean(dim=1).round(3))
else:
    print("Not enough data to reshape into 40x10.")

# Unsqueeze and squeeze
unsqueezed = t_covers.unsqueeze(0)
print(f"Unsqueezed (adds dimension at 0): shape {unsqueezed.shape}")
squeezed = unsqueezed.squeeze()
print(f"Squeezed back: shape {squeezed.shape}")

# Transpose (for 2D tensors)
if t_centroids.dim() == 2:
    transposed = t_centroids.T
    print(f"Transposed centroids shape: {transposed.shape}")

# =============================================================================
# 9. GPU / CPU (if available)
# =============================================================================
print("\n" + "=" * 60)
print("GPU / CPU")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    gpu_tensor = t_covers.cuda()
    print(f"Tensor on GPU: {gpu_tensor.device}")
    cpu_back = gpu_tensor.cpu()
    print(f"Back to CPU: {cpu_back.device}")
else:
    print("❌ GPU not available, using CPU only.")

# =============================================================================
# 10. Type conversion
# =============================================================================
print("\n" + "=" * 60)
print("TYPE CONVERSION")
print("=" * 60)

float_tensor = t_covers.clone()
int_tensor = float_tensor.int()
bool_tensor = float_tensor > 0.5
print(f"Original dtype: {float_tensor.dtype}")
print(f"As integer: {int_tensor.dtype} (values: {int_tensor[:5]})")
print(f"As boolean (cover > 0.5): {bool_tensor[:5]}")

# =============================================================================
# 11. Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
✅ Successfully loaded GeoJSON data
✅ Exported all properties to '{csv_file}'
✅ Extracted N_Parcel, Parcel_Cover, and polygon centroids
✅ Created PyTorch tensors and performed:
   - Tensor properties (shape, dtype, numel)
   - Indexing and slicing with boolean masks
   - Mathematical operations (sum, mean, std, elementwise)
   - Reshaping, unsqueeze, squeeze, transpose
   - GPU/CPU transfer (if available)
   - Type conversions

🔜 Next: You can now use these tensors for machine learning tasks!
""")