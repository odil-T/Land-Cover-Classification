from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


image_path = "../data/OpenEarthMap/OpenEarthMap_wo_xBD/viru/images/viru_47.tif"
mask_path = "../data/OpenEarthMap/OpenEarthMap_wo_xBD/viru/labels/viru_47.tif"

colormap = [
    (0, 0, 0),        # Black for the first class
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (0, 255, 255),    # Cyan
    (255, 192, 203)   # Pink
]

image = Image.open(image_path)
image = np.array(image)

mask = Image.open(mask_path)
mask = np.array(mask)

output_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for class_id, color in enumerate(colormap):
    output_mask[mask == class_id] = color

plt.imshow(image)
plt.show()

plt.imshow(output_mask)
plt.show()