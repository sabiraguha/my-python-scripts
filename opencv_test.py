import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

# Create a simple blue image
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:, :, 0] = 255  # Blue channel

# Draw a green circle
cv2.circle(img, (300, 200), 80, (0, 255, 0), -1)

# Save the image
cv2.imwrite("opencv_result.jpg", img)
print("✅ Image saved: opencv_result.jpg")
print("OpenCV test completed successfully!")