#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_opencv_satellite.py
Objective: Learn OpenCV for satellite image processing
Author: Amo
Date: March 2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Create folders
os.makedirs('screenshots', exist_ok=True)
os.makedirs('opencv_output', exist_ok=True)

print("=" * 60)
print("OPENCV FOR SATELLITE IMAGE PROCESSING")
print("=" * 60)
print(f"OpenCV version: {cv2.__version__}")
print("=" * 60)

# -------------------------------------------------------------------
# 1. CREATE SIMULATED SATELLITE IMAGES (CORRECTED VERSION)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. CREATING SIMULATED SATELLITE IMAGES")
print("=" * 60)

def create_simulated_satellite_image(size=256):
    """Create a simulated satellite image with fields, forests, water"""
    # Create base image
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add sky/background
    img[:, :, 0] = 135  # Blue channel
    img[:, :, 1] = 206  # Green channel
    img[:, :, 2] = 235  # Red channel
    
    # Add forest (dark green) - random patches
    forest = np.random.rand(size//2, size//2) > 0.7
    for i in range(forest.shape[0]):
        for j in range(forest.shape[1]):
            if forest[i, j]:
                x, y = i*2, j*2
                if x+2 <= size and y+2 <= size:
                    img[x:x+2, y:y+2] = [34, 139, 34]  # Forest green
    
    # Add agricultural fields (different shades of green/brown)
    colors = [
        [124, 252, 0],   # Lawn green
        [154, 205, 50],  # Yellow green
        [107, 142, 35],  # Olive drab
        [160, 82, 45],   # Sienna (bare soil)
        [210, 180, 140]  # Tan (dry vegetation)
    ]
    
    # Create field patches (40x40 pixels)
    for i in range(0, size, 40):
        for j in range(0, size, 40):
            # Randomly choose a color from the list
            color_idx = np.random.randint(0, len(colors))
            color = colors[color_idx]
            
            # Ensure we don't go out of bounds
            end_i = min(i+38, size)
            end_j = min(j+38, size)
            img[i:end_i, j:end_j] = color
    
    # Add river (blue line)
    cv2.line(img, (50, 50), (200, 200), (255, 255, 0), 5)
    cv2.line(img, (200, 200), (350, 350), (255, 255, 0), 5)
    
    # Add clouds (white circles) - random
    for _ in range(10):
        x = np.random.randint(20, size-20)
        y = np.random.randint(20, size-20)
        radius = np.random.randint(10, 25)
        cv2.circle(img, (x, y), radius, (255, 255, 255), -1)
    
    return img

# Create simulated image
sim_img = create_simulated_satellite_image(400)
cv2.imwrite('opencv_output/simulated_satellite.jpg', sim_img)
print("✅ Simulated satellite image saved to opencv_output/simulated_satellite.jpg")

# Display
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB))
plt.title('Simulated Satellite Image')
plt.axis('off')
plt.savefig('screenshots/08_simulated_satellite.png', dpi=150)
plt.show()

# -------------------------------------------------------------------
# 2. BASIC OPENCV OPERATIONS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. BASIC OPENCV OPERATIONS")
print("=" * 60)

# Load image
img = cv2.imread('opencv_output/simulated_satellite.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2.1 Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('opencv_output/01_grayscale.jpg', gray)

# 2.2 Apply Gaussian blur
blurred = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imwrite('opencv_output/02_blurred.jpg', blurred)

# 2.3 Edge detection (Canny)
edges = cv2.Canny(gray, 50, 150)
cv2.imwrite('opencv_output/03_edges.jpg', edges)

# 2.4 Image thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('opencv_output/04_threshold.jpg', thresh)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Blur')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(thresh, cmap='gray')
plt.title('Thresholding')
plt.axis('off')

plt.tight_layout()
plt.savefig('screenshots/08_basic_operations.png', dpi=150)
plt.show()

print("✅ Basic operations saved")

# -------------------------------------------------------------------
# 3. SEGMENTATION - K-MEANS CLUSTERING
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. IMAGE SEGMENTATION (K-Means)")
print("=" * 60)

# Reshape image for k-means
pixels = img.reshape((-1, 3))
pixels = np.float32(pixels)

# Apply k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
k = 5  # Number of clusters (land cover types)
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back to image
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(img.shape)

cv2.imwrite('opencv_output/05_kmeans_segmentation.jpg', segmented_image)

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title(f'K-Means Segmentation (k={k})')
plt.axis('off')

plt.subplot(1, 3, 3)
# Show cluster labels
label_image = labels.reshape(img.shape[:2])
plt.imshow(label_image, cmap='tab20')
plt.title('Cluster Labels')
plt.axis('off')
plt.colorbar(label='Cluster ID')

plt.tight_layout()
plt.savefig('screenshots/08_kmeans_segmentation.png', dpi=150)
plt.show()

print(f"✅ K-means segmentation with k={k} completed")

# -------------------------------------------------------------------
# 4. MORPHOLOGICAL OPERATIONS
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. MORPHOLOGICAL OPERATIONS")
print("=" * 60)

# Create a binary image from edges
binary = edges.copy()
binary[binary > 0] = 255

# Define kernel
kernel = np.ones((5,5), np.uint8)

# Erosion
erosion = cv2.erode(binary, kernel, iterations=1)
cv2.imwrite('opencv_output/06_erosion.jpg', erosion)

# Dilation
dilation = cv2.dilate(binary, kernel, iterations=1)
cv2.imwrite('opencv_output/07_dilation.jpg', dilation)

# Opening (erosion then dilation)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cv2.imwrite('opencv_output/08_opening.jpg', opening)

# Closing (dilation then erosion)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('opencv_output/09_closing.jpg', closing)

# Display
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(binary, cmap='gray')
plt.title('Binary (from edges)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.tight_layout()
plt.savefig('screenshots/08_morphological_operations.png', dpi=150)
plt.show()

print("✅ Morphological operations saved")

# -------------------------------------------------------------------
# 5. FEATURE DETECTION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. FEATURE DETECTION")
print("=" * 60)

# 5.1 Corner detection
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
if corners is not None:
    corners = np.int0(corners)
    img_corners = img_rgb.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img_corners, (x, y), 5, (255, 0, 0), -1)
    
    # 5.2 Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img_rgb.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    
    # Display
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_corners)
    plt.title(f'Corners detected: {len(corners)}')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_contours)
    plt.title(f'Contours detected: {len(contours)}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('screenshots/08_feature_detection.png', dpi=150)
    plt.show()
    
    print("✅ Feature detection saved")
else:
    print("⚠️ No corners detected")

# -------------------------------------------------------------------
# 6. NDVI SIMULATION (Vegetation Index)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. SIMULATING NDVI (Vegetation Index)")
print("=" * 60)

# Simulate Red and NIR bands
# In real satellites: NDVI = (NIR - Red) / (NIR + Red)
red = img[:, :, 2].astype(np.float32)  # Red band
nir = img[:, :, 1].astype(np.float32) * 1.5  # Simulated NIR (near-infrared)

# Calculate NDVI
ndvi = (nir - red) / (nir + red + 1e-8)
ndvi = np.clip(ndvi, -1, 1)

# Normalize for display
ndvi_display = ((ndvi + 1) * 127.5).astype(np.uint8)

cv2.imwrite('opencv_output/10_ndvi.jpg', ndvi_display)

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original RGB')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title('NDVI (Vegetation Index)')
plt.colorbar(label='NDVI')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.hist(ndvi.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('NDVI Distribution')
plt.xlabel('NDVI')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('screenshots/08_ndvi_simulation.png', dpi=150)
plt.show()

print("✅ NDVI simulation saved")

# -------------------------------------------------------------------
# 7. APPLY TO REAL SATELLITE DATA (if available)
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. APPLYING TO REAL SATELLITE DATA")
print("=" * 60)

# Check if we have real satellite images
real_images = []
for file in os.listdir('.'):
    if file.endswith(('.jpg', '.png', '.tif')) and 'satellite' in file.lower():
        real_images.append(file)

if real_images:
    print(f"Found real images: {real_images}")
    for img_file in real_images[:1]:  # Process first one
        real_img = cv2.imread(img_file)
        if real_img is not None:
            real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
            real_edges = cv2.Canny(real_gray, 50, 150)
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
            plt.title('Real Satellite')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(real_gray, cmap='gray')
            plt.title('Grayscale')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(real_edges, cmap='gray')
            plt.title('Edges')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('screenshots/08_real_satellite_processing.png', dpi=150)
            plt.show()
        else:
            print(f"⚠️ Could not read image: {img_file}")
else:
    print("No real satellite images found in current directory")
    print("You can add your own images later")

# -------------------------------------------------------------------
# 8. SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. SUMMARY")
print("=" * 60)

print("""
✅ OpenCV Skills Learned:

1. Basic Operations:
   - Reading/writing images
   - Color conversion (BGR, RGB, grayscale)
   - Gaussian blur
   - Edge detection (Canny)
   - Thresholding

2. Segmentation:
   - K-means clustering for land cover classification

3. Morphological Operations:
   - Erosion, Dilation
   - Opening, Closing

4. Feature Detection:
   - Corner detection
   - Contour detection

5. Vegetation Indices:
   - NDVI simulation

📁 Output saved in:
   - screenshots/ (PNG visualizations)
   - opencv_output/ (JPG results)

🔜 Next steps:
   1. Apply to real PASTIS images
   2. Extract features for machine learning
   3. Combine with LSTM models
""")

print("\n" + "=" * 60)
print("OPENCV TUTORIAL COMPLETE! 🖼️")
print("=" * 60)