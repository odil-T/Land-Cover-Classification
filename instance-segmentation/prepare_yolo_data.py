"""
Saves the images and labels from SpaceNet-v2 dataset to other directories to be used by YOLO.
The images and labels are split 80-20 into train and validation sets.
The images are converted from .tif to .png format.
The labels are converted from .geojson to .txt format suitable for YOLO.
"""

import os
import geojson
import numpy as np
from PIL import Image
from osgeo import gdal
from sklearn.model_selection import train_test_split


image_height = 650
image_width = 650


def create_directory(directory_path):
    """
    Creates a directory in the specified directory path.
    Raises an error if the directory already exists.

    Args:
        directory_path (str): The path of the directory for which to create.
    """

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        raise FileExistsError(f"Directory '{directory_path}' already exists. Delete it if you wish to prepare new YOLO data.")


def geojson2yolo(geojson_path, image_path, set_name):
    """
    Converts the building instance polygons from geojson to YOLO format.
    Saves the annotations as .txt files in a separate directory.
    The annotations are saved in either `train` or `val` directory, depending on the value provided to `set_name` argument.

    Args:
        geojson_path (str): Path to the .geojson label file.
        image_path (str): Path to the .tif image file that corresponds to the .geojson file.
        set_name (str): Indicates whether the .geojson file belongs to `train` or `val` set. No other values are accepted.
    """

    with open(geojson_path) as f:
        gj = geojson.load(f)

    num_buildings = len(gj["features"])

    txt_file_name = geojson_path.split('/')[-1].replace('buildings_', '').replace('.geojson', '.txt')

    if set_name != "train" and set_name != "val":
        raise Exception("Please provide either `train` or `val` values for the `set_name` argument.")

    txt_file_path = f"datasets/{set_name}_yolo/labels/{txt_file_name}"

    with open(txt_file_path, "w") as f:
        if num_buildings > 0:
            gdal_image = gdal.Open(image_path)

            pixel_width, pixel_height = gdal_image.GetGeoTransform()[1], gdal_image.GetGeoTransform()[5]
            originX, originY = gdal_image.GetGeoTransform()[0], gdal_image.GetGeoTransform()[3]

            for i in range(num_buildings):
                f.write("0 ")

                points = gj["features"][i]["geometry"]["coordinates"][0]
                if len(points) == 1:
                    points = points[0]

                for j in range(len(points)):

                    if isinstance(points[j][0], list):
                        points[j][0] = points[j][0][0]

                    if isinstance(points[j][1], list):
                        points[j][1] = points[j][1][1]

                    point_x = int(round((points[j][0] - originX) / pixel_width)) / image_width  # normalizing
                    point_y = int(round((points[j][1] - originY) / pixel_height)) / image_height

                    f.write(f"{point_x:.3f} {point_y:.3f} ")

                f.write("\n")


def tif2png(image_path, set_name):
    """
    Converts the .tif images to .png format in a separate directory.
    The iamges are saved in either `train` or `val` directory, depending on the value provided to `set_name` argument.

    Args:
        image_path (str): Path to the .tif image file.
        set_name (str): Indicates whether the .geojson file belongs to `train` or `val` set. No other values are accepted.
    """

    image_file_name = image_path.split('/')[-1].replace('RGB-PanSharpen_', '').replace('.tif', '.png')

    if set_name != "train" and set_name != "val":
        raise Exception("Please provide either `train` or `val` values for the `set_name` argument.")

    image_file_path = f"datasets/{set_name}_yolo/images/{image_file_name}"

    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)

    scaled_bands = []
    for band_index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(band_index).ReadAsArray()
        scaled_band = (band - band.min()) / (band.max() - band.min())
        scaled_bands.append(scaled_band)
    image_normalized = np.stack(scaled_bands, axis=-1)

    image = Image.fromarray((image_normalized * 255).astype(np.uint8))
    image.save(image_file_path)


aoi_root_dir_paths = [
    "datasets/SN2_buildings_train_AOI_2_Vegas/AOI_2_Vegas_Train",
    "datasets/SN2_buildings_train_AOI_3_Paris/AOI_3_Paris_Train",
    "datasets/SN2_buildings_train_AOI_4_Shanghai/AOI_4_Shanghai_Train",
    "datasets/SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train",
]

# Iterating for every AOI.
for aoi_root_dir_path in aoi_root_dir_paths:

    # Obtaining geojson label file names and splitting them into train and validation sets.
    label_file_names = os.listdir(f"{aoi_root_dir_path}/geojson/buildings")
    y_train, y_val = train_test_split(label_file_names, test_size=0.2, shuffle=True, random_state=42)

    # Iterating for both train and validation sets.
    for set_, set_name in zip([y_train, y_val], ["train", "val"]):

        # Creating directories to save the reformatted images and labels to be used later for training the YOLO model.
        create_directory(f"datasets/{set_name}_yolo")
        create_directory(f"datasets/{set_name}_yolo/images")
        create_directory(f"datasets/{set_name}_yolo/labels")

        # Iterating for every .tif image and .geojson file for each set.
        for label_file_name in set_:
            geojson_path = f'{aoi_root_dir_path}/geojson/buildings/{label_file_name}'
            image_path = f'{aoi_root_dir_path}/RGB-PanSharpen/{label_file_name.replace("buildings", "RGB-PanSharpen").replace(".geojson", ".tif")}'

            # Saving the images and labels in the previously created directories.
            geojson2yolo(geojson_path, image_path, set_name)
            tif2png(image_path, set_name)
