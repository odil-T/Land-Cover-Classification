"""
Converts the labels of certain categories from the VALID dataset to YOLO format.
"""

import json
import os
import re


desired_category_names = ["building", "smallvehicle", "largevehicle"]
labels_dir_path = "data/VALID/label/label"


def convert_xywh_to_yolo(xywh, image_width, image_height):
    """
    Converts XYWH (top-left) bounding box format to YOLO format.

    Args:
        xywh (list): The bounding box to be converted. Needs to be in XYWH format.
        image_width (int): Width of the image in which the bounding box is located for reference.
        image_height (int): Height of the image in which the bounding box is located for reference.

    Returns:
        tuple: Normalized YOLO format bounding box. The values in the tuple are strings.
    """

    x, y, w, h = xywh

    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height

    relative_width = w / image_width
    relative_height = h / image_height

    bounding_box = tuple(format(value, ".6f") for value in (x_center, y_center, relative_width, relative_height))

    return bounding_box


for filename in os.listdir(labels_dir_path):
    label_file_path = os.path.join(labels_dir_path, filename)

    with open(label_file_path, 'r') as file:
        label_file_metadata = json.load(file)

    file_name = re.search(r"[^/]+$", label_file_metadata["file_name"]).group(0)  # img_1_0_1552039977070010400.png
    image_width = label_file_metadata["width"]
    image_height = label_file_metadata["height"]

    # Saving annotations in YOLO format in a new txt file
    with open(os.path.join("data/VALID/YOLO_format_labels", file_name.replace(".png", ".txt")), "w") as f:
        for annotation in label_file_metadata["detection"]:  # each annotation is a dictionary

            # Filter annotations of desired categories
            if annotation["category_name"] in desired_category_names:
                bounding_box = convert_xywh_to_yolo(annotation["hbbox"], image_width, image_height)

                f.write(str(annotation["category_id"]) + " ")
                for a in bounding_box[:-1]:
                    f.write(a + " ")
                f.write(bounding_box[-1] + "\n")
