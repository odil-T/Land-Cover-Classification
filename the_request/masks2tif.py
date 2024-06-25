"""
This file converts the mask patches into a single large mask as a .tif file.
"""

from osgeo import gdal, gdal_array
import numpy as np
import os


# Parameters
masks_dir = 'Tashkent_images/masks'
original_tif_path = 'Tashkent_images/j_10_030.tif'
output_tif_path = 'Tashkent_images/j_10_030_mask.tif'
patch_size = 650


def combine_patches_to_tiff(masks_dir, original_tif_path, output_tif_path, patch_size):
    # Open the original TIFF file to get its metadata
    original_dataset = gdal.Open(original_tif_path, gdal.GA_ReadOnly)
    if not original_dataset:
        raise FileNotFoundError(f"Failed to open file: {original_tif_path}")

    x_size = original_dataset.RasterXSize
    y_size = original_dataset.RasterYSize
    projection = original_dataset.GetProjection()
    geotransform = original_dataset.GetGeoTransform()

    # Create an empty array for the full mask image
    full_mask = np.zeros((y_size, x_size), dtype=np.uint8)

    # Iterate over the patches and place them in the correct position in the full mask
    for y in range(0, y_size, patch_size):
        for x in range(0, x_size, patch_size):
            patch_filename = os.path.join(masks_dir, f'patch_{x}_{y}.png')
            if not os.path.exists(patch_filename):
                continue

            # Read the patch
            patch = gdal_array.LoadFile(patch_filename)

            # Get the dimensions of the patch
            patch_height, patch_width = patch.shape

            # Place the patch in the correct position in the full mask array
            full_mask[y:y + patch_height, x:x + patch_width] = patch

    # Create a new TIFF file with the same dimensions and geotransform as the original
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_tif_path, x_size, y_size, 1, gdal.GDT_Byte)
    out_dataset.SetProjection(projection)
    out_dataset.SetGeoTransform(geotransform)

    # Write the full mask to the new TIFF file
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(full_mask)

    # Close datasets
    original_dataset = None
    out_dataset = None
    print(f"Stitched mask saved to {output_tif_path}")


# Run the function to combine patches into one large TIFF
combine_patches_to_tiff(masks_dir, original_tif_path, output_tif_path, patch_size)
