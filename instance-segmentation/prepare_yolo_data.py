"""
Saves the images and labels from SpaceNet-v2 dataset to other directories to be used by YOLO.
The images and labels are split 80-20 into train and validation sets.
The images are converted from .tif to .png format.
The labels are converted from .geojson to .txt format suitable for YOLO.
"""

import os
import json
import geojson
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# iterate for every AOI

image_height = 650
image_width = 650


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        raise FileExistsError(f"Directory '{directory_path}' already exists. Delete it if you wish to prepare new YOLO data.")


def geojson2yolo(geojson_path, image_path, set_name):
    """


    Args:
        geojson_path:
        image_path:
    """

    with open(geojson_path) as f:
        gj = geojson.load(f)

    num_buildings = len(gj["features"])

    txt_file_name = geojson_path.split('/')[-1].replace('buildings_', '').replace('.geojson', '.txt')

    if set_name == "train":
        txt_file_path = f"datasets/train_yolo/labels/{txt_file_name}"
    elif set_name == "val":
        txt_file_path = f"datasets/val_yolo/labels/{txt_file_name}"
    else:
        raise Exception("Please provide either `train` or `val` values for the `set_name` argument.")


    with open(txt_file_path, "w") as f:
        if num_buildings > 0:
            gdal_image = gdal.Open(image_path)  # must be image file path

            pixel_width, pixel_height = gdal_image.GetGeoTransform()[1], gdal_image.GetGeoTransform()[5]
            originX, originY = gdal_image.GetGeoTransform()[0], gdal_image.GetGeoTransform()[3]

            for i in range(num_buildings):
                f.write("0 ")

                points = gj["features"][i]["geometry"]["coordinates"][0]
                if len(points) == 1:
                    points = points[0]

                for j in range(len(points)):
                    point_x = int(round((points[j][0] - originX) / pixel_width)) / image_width  # normalizing
                    point_y = int(round((points[j][1] - originY) / pixel_height)) / image_height

                    f.write(f"{point_x:.3f} {point_y:.3f} ")

                f.write("\n")


def tif2png():
    pass


aoi_root_dir_path = "datasets/SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train"

label_file_names = os.listdir(f"{aoi_root_dir_path}/geojson/buildings")
y_train, y_val = train_test_split(label_file_names, test_size=0.2, shuffle=True, random_state=42)


for set_, set_name in zip([y_train, y_val], ["train", "val"]):

    create_directory(f"datasets/{set_name}_yolo")
    create_directory(f"datasets/{set_name}_yolo/images")
    create_directory(f"datasets/{set_name}_yolo/labels")

    for label_file_name in set_:
        geojson_path = f'{aoi_root_dir_path}/geojson/buildings/{label_file_name}'
        image_path = f'{aoi_root_dir_path}/RGB-PanSharpen/{label_file_name.replace("buildings", "RGB-PanSharpen").replace(".geojson", ".tif")}'

        geojson2yolo(geojson_path, image_path, set_name)
