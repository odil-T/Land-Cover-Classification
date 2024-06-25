"""
This file divides a large .tif file into smaller patches. These patches are saved as .png files.

You may specify the patch size as well as the directory to save the images.
"""

import os
import numpy as np
from osgeo import gdal
from PIL import Image


# Parameters
input_file = 'Tashkent_images/j_10_030.tif'
patch_size = 650
output_dir = 'Tashkent_images/patches'


def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def save_patch_as_png(output_dir, img_array, x, y, dataset):
    patch_filename = os.path.join(output_dir, f'patch_{x}_{y}.png')

    # Normalize and scale the image array
    scaled_bands = []
    for band_index in range(1, dataset.RasterCount + 1):
        band = img_array[band_index - 1]
        scaled_band = (band - band.min()) / (band.max() - band.min())
        scaled_bands.append(scaled_band)
    image_normalized = np.stack(scaled_bands, axis=-1)

    # Convert to an image and save as PNG
    image = Image.fromarray((image_normalized * 255).astype(np.uint8))
    image.save(patch_filename)
    print(f"Saved patch to {patch_filename}")


def divide_image_to_patches(input_file, patch_size, output_dir):
    create_output_directory(output_dir)

    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    if not dataset:
        raise FileNotFoundError(f"Failed to open file: {input_file}")

    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize

    for y in range(0, y_size, patch_size):
        for x in range(0, x_size, patch_size):
            x_offset = x
            y_offset = y
            x_window_size = min(patch_size, x_size - x_offset)
            y_window_size = min(patch_size, y_size - y_offset)

            img_array = []
            for band_index in range(1, dataset.RasterCount + 1):
                band = dataset.GetRasterBand(band_index).ReadAsArray(x_offset, y_offset, x_window_size, y_window_size)
                img_array.append(band)
            img_array = np.array(img_array)

            save_patch_as_png(output_dir, img_array, x, y, dataset)

    dataset = None
    print("Image division into patches completed.")


# Run the function to divide the image, predict masks, and save patches as PNG
divide_image_to_patches(input_file, patch_size, output_dir)
