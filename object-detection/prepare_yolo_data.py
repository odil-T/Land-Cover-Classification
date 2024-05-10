"""
Copies and saves the images and labels from VALID to other directories to be used by YOLO.
The images and labels are split into train, validation, and test sets in 70-20-10 proportions, respectively.
The images are taken from the original VALID directory.
The labels are taken from the directory to which the labels were saved in YOLO format by the `convert_labels.py` file.
"""

import os
import shutil
from sklearn.model_selection import train_test_split


def find_file_in_directory(root_directory, filename):
    """
    Finds a file from a root directory and returns its file path. The file is searched by its file name.

    Args:
        root_directory (str): The directory in which to search the file.
        filename (str): The name of the file for which to search.

    Returns:
        str: The file path of the file name.
    """

    for root, dirs, files in os.walk(root_directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


txt_label_file_names = os.listdir("data/VALID/YOLO_format_labels")

y_temp, y_val = train_test_split(txt_label_file_names, test_size=0.2, shuffle=True, random_state=42)
y_train, y_test = train_test_split(y_temp, test_size=0.125, shuffle=True, random_state=42)

for set_, set_name in zip([y_train, y_val, y_test], ["train", "val", "test"]):
    for file_name in set_:

        # Copying labels
        label_source_path = os.path.join("data/VALID/YOLO_format_labels", file_name)
        label_destination_path = os.path.join(f"data/{set_name}_yolo/labels", file_name)
        shutil.copy2(label_source_path, label_destination_path)

        # Copying images
        image_source_path = str(find_file_in_directory("data/VALID/images/images",
                                                   file_name.replace(".txt", ".png")))
        image_destination_path = os.path.join(f"data/{set_name}_yolo/images", file_name.replace(".txt", ".png"))
        shutil.copy2(image_source_path, image_destination_path)
