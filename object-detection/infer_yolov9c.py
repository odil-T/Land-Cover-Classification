"""
Displays test set images with their ground truth and predicted bounding boxes for comparison.
The predicted bounding boxes are produced by a pretrained model.
The samples are taken randomly.
You may specify the number of samples to display and the model to use.
"""

import os
import cv2
import random
from matplotlib import pyplot as plt
from ultralytics import YOLO


# Specify number of samples to display from the test set
num_samples = 5

model = YOLO("")


def display_results(n, model):
    """
    Displays random N number of images with ground truth and predicted bounding boxes from the test set of VALID.

    Args:
        n (int): The number of random samples from the test set to display.
        model: YOLO model to use for object detection.
    """

    test_image_paths = [os.path.join(f"datasets/test_yolo/images/{image_name}")
                        for image_name in os.listdir("datasets/test_yolo/images")]
    random_n_image_paths = random.sample(test_image_paths, n)
    random_n_label_paths = [image_path.replace("images", "labels").replace(".png", ".txt")
                            for image_path in random_n_image_paths]

    results = model(random_n_image_paths)

    for i, result in enumerate(results):
        original_image = cv2.imread(random_n_image_paths[i])
        ground_truth_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        predicted_image = ground_truth_image.copy()

        # Drawing ground truth bounding boxes
        image_height, image_width = ground_truth_image.shape[:-1]

        with open(random_n_label_paths[i], "r") as f:
            for line in f.readlines():
                center_x, center_y, width, height = [float(coord) for coord in line.split()[1:]]

                # Convert YOLO format to coordinates of top-left and bottom-right corners
                x1 = int((center_x - width / 2) * image_width)
                y1 = int((center_y - height / 2) * image_height)
                x2 = int((center_x + width / 2) * image_width)
                y2 = int((center_y + height / 2) * image_height)

                cv2.rectangle(ground_truth_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Drawing predicted bounding boxes
        boxes = result.boxes
        boxes = boxes.xyxy.tolist()

        for box in boxes:
            box = [int(coord) for coord in box]
            x1, y1, x2, y2 = box
            cv2.rectangle(predicted_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(ground_truth_image)
        axs[0].axis('off')
        axs[0].set_title("Ground Truth BB")

        axs[1].imshow(predicted_image)
        axs[1].axis('off')
        axs[1].set_title("Predicted BB")

        plt.show()


display_results(num_samples, model)