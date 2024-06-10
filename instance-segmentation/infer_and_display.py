"""
Displays validation set images with their ground truth and predicted instance masks for comparison.
The predicted instance masks are produced by a pretrained model.
The samples are taken randomly.
You may specify the number of samples to display.
"""

import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO


# Specify number of samples to display from the validation set
num_samples = 5

model = YOLO("yolov8n-seg")

blacklisted_colors = [
    [0, 0, 0],        # Black           --- Background
    [128, 0, 0],      # Maroon          --- Bareland
    [0, 255, 36],     # Light Green     --- Rangeland
    [148, 148, 148],  # Grey            --- Developed space
    [255, 255, 255],  # White           --- Road
    [34, 97, 38],     # Dark Green      --- Tree
    [0, 69, 255],     # Blue            --- Water
    [75, 181, 73],    # Green           --- Agriculture land
    [222, 31, 7],     # Red             ---	Building
]


def random_color(blacklisted_colors):
    """
    Generates a random color different from the blacklisted colors.

    Args:
        blacklisted_colors (list): A list of colors that should not be generated.

    Returns:
        list: The randomly generated color.
    """
    color = None
    while color is None or color in blacklisted_colors:
        color = np.random.randint(0, 255, size=3).tolist()
    return color


def display_results(n, model):
    """
    Displays random N number of images with ground truth and predicted instance masks from the validation set of
    SpaceNet-v2 dataset.

    Args:
        n (int): The number of random samples from the validation set to display.
        model: YOLO model to use for instance segmentation.
    """

    val_image_paths = [os.path.join(f"datasets/val_yolo/images/{image_name}")
                        for image_name in os.listdir("datasets/val_yolo/images")]
    random_n_image_paths = random.sample(val_image_paths, n)
    random_n_label_paths = [image_path.replace("images", "labels").replace(".png", ".txt")
                            for image_path in random_n_image_paths]

    results = model(random_n_image_paths)

    for i, result in enumerate(results):
        original_image = cv2.imread(random_n_image_paths[i])
        image_height, image_width = original_image.shape[:2]

        ground_truth_image = np.zeros((image_height, image_width), dtype=int)
        predicted_image = ground_truth_image.copy()


        # Drawing ground truth instance masks
        with open(random_n_label_paths[i], "r") as f:
            for line in f.readlines():
                yolo_polygon = [float(coord) for coord in line.split()[1:]]

                # Converting to polygon format that OpenCV can understand
                polygon = []
                for i in range(0, len(yolo_polygon), 2):
                    polygon.append([int(yolo_polygon[i] * image_width), int(yolo_polygon[i + 1] * image_height)])
                polygon = np.array(polygon)

                cv2.fillPoly(ground_truth_image, [polygon], 255)


        # Drawing predicted instance masks
        polygons = result.masks.cpu().xy

        for polygon in polygons:
            polygon = np.array(polygon).astype(np.int32)

            color = random_color(blacklisted_colors)
            blacklisted_colors.append(color)

            cv2.fillPoly(predicted_image, [polygon], color)


        fig, axs = plt.subplots(1, 3, figsize=(10, 5))

        axs[0].imshow(original_image)
        axs[0].axis('off')
        axs[0].set_title("Original Image")

        axs[1].imshow(ground_truth_image)
        axs[1].axis('off')
        axs[1].set_title("Ground Truth Instance Masks")

        axs[2].imshow(predicted_image)
        axs[2].axis('off')
        axs[2].set_title("Predicted Instance Masks")

        plt.show()


display_results(num_samples, model)
