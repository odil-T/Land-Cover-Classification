import rasterio
import numpy as np
import matplotlib.pyplot as plt


def view_tif_as_rgb(tif_path):
    """
    View a TIFF file as an RGB image.

    Parameters:
        tif_path (str): Path to the TIFF file.
    """
    with rasterio.open(tif_path) as src:
        # Read the RGB bands. Assuming the TIFF has 3 bands and they correspond to RGB.
        r = src.read(1)  # Red band
        g = src.read(2)  # Green band
        b = src.read(3)  # Blue band

        # Stack the bands into an RGB image
        rgb = np.dstack((r, g, b))

        # Display the image
        plt.imshow(rgb)
        plt.title("RGB Image")
        plt.axis('off')  # Hide the axis
        plt.show()


# Example call to view the image
view_tif_as_rgb('masks/patch_0_0.tif')
