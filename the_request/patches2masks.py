"""
I changed the instance mask to return None if no predictions are found.
"""

import os
import numpy as np
from osgeo import gdal
from PIL import Image
import torch
from utils import *
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor
from transformers import SegformerForSemanticSegmentation
from ultralytics import YOLO


patches_dir = 'Tashkent_images/patches'
masks_dir = 'Tashkent_images/masks'



def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


num_classes = 9
device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify path of image to infer
image_path = r"img1.png"

# Load SegFormer model
segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2",
                                                         num_labels=num_classes).to(device)
checkpoint = torch.load("best_models/segformer_sem_seg_checkpoint_epoch35.pt")
segformer.load_state_dict(checkpoint["model_state_dict"])

# Load YOLO model
yolo = YOLO("best_models/best.pt")


target_size = 650

colormap = [
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


def preprocess_image(image_path, target_size):
    """
    Resizes the image to the specified target size.

    Args:
        image_path (str): The path of the image to load.
        target_size (int): The height and width of the image to resize to. Since both dimensions are the same, this results
        in a square image.

    Returns:
        numpy.ndarray: The resized image.
    """

    image = Image.open(image_path)
    image_width, image_height = image.size

    if image_width >= target_size and image_height >= target_size:
        image = np.array(crop_center(image, target_size))
    else:
        image = resize_and_pad(np.array(image), target_size, False)

    return image[:, :, :3]


def infer_segformer(image_path, model, target_size, colormap):
    """
    Performs semantic segmentation on an image.

    Args:
        image_path (str): The path of the image to infer.
        model: PyTorch SegFormer model to use for inference.
        target_size (int): The height and width of the output mask.
        colormap (list): The colormap to indicate the colors of different classes.

    Returns:
        numpy.ndarray: The inferred semantic mask.
    """

    colormap = colormap[:num_classes]

    image = preprocess_image(image_path, target_size)
    image = ToTensor()(image).to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        pred_logits = model(image).logits  # output: torch.Tensor (1, 9, height/4, width/4)
        upsampled_logits = nn.functional.interpolate(pred_logits,
                                                     size=(target_size, target_size), mode="bilinear",
                                                     align_corners=False)  # output: torch.Tensor (1, 9, height, width)
        pred = upsampled_logits.argmax(dim=1)
        pred = pred.squeeze(0)
        pred = pred.cpu()

    # Predicted mask
    pred_mask_output = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colormap):
        pred_mask_output[pred == class_id] = color

    return pred_mask_output


def infer_yolo(image_path, model, target_size, blacklisted_colors):
    """
    Performs instance segmentation on an image.

    Args:
        image_path (str): The path of the image to infer.
        model: YOLO model to use for inference.
        target_size (int): The height and width of the output mask.
        blacklisted_colors (list): The list of colors which the instance masks cannot use.

    Returns:
        numpy.ndarray: The inferred instance mask.
    """

    image = preprocess_image(image_path, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(image)

    if results[0].masks is None:
        return None

    for result in results:
        predicted_image = np.zeros(image.shape, dtype=int)

        # Drawing predicted instance masks
        polygons = result.masks.cpu().xy

        for polygon in polygons:
            polygon = np.array(polygon).astype(np.int32)

            color = random_color(blacklisted_colors)
            blacklisted_colors.append(color)

            try:
                cv2.fillPoly(predicted_image, [polygon], color)

            except cv2.error:
                print(f"\033[91mWarning: cv2.error encountered. Skipping instance segmentation for patch {image_path}.\033[0m")
                return None

        return predicted_image


def smoothen_borders(mask):
    smoothed_mask = np.zeros_like(mask)

    for color in colormap:
        feature_mask = cv2.inRange(mask, np.array(color), np.array(color))

        contours, _ = cv2.findContours(feature_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.0001 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
            approx = cv2.approxPolyDP(contour, epsilon, True)

            cv2.drawContours(smoothed_mask, [approx], -1, color, thickness=cv2.FILLED)

    return smoothed_mask


def infer_both(image_path):
    """
    Displays the original image and the panoptic mask by overlaying the instance masks on top of the semantic mask.

    Args:
        image_path (str): The path of the image to infer.
    """

    semantic_mask = infer_segformer(image_path, segformer, target_size, colormap)
    instance_mask = infer_yolo(image_path, yolo, target_size, colormap)

    if instance_mask is not None:
        # Merging the masks
        black_mask = np.all(instance_mask == [0, 0, 0], axis=-1)
        overlay_mask = ~black_mask
        overlay_mask = overlay_mask.astype(np.uint8)
        overlay_mask = np.expand_dims(overlay_mask, axis=-1)
        merged_mask = semantic_mask * (1 - overlay_mask) + instance_mask * overlay_mask

        return smoothen_borders(merged_mask)

    return smoothen_borders(semantic_mask)


def save_rgb_image_pillow(rgb_array, output_path):
    image = Image.fromarray(rgb_array, 'RGB')
    image.save(output_path)


create_output_directory(masks_dir)

for patch_name in os.listdir(patches_dir):
    patch_path = f"{patches_dir}/{patch_name}"
    print(f"Now processing {patch_path}", end='')

    mask = infer_both(patch_path)
    mask_path = f"{masks_dir}/{patch_name}"

    save_rgb_image_pillow(mask, mask_path)
    print(f"Saved mask to {mask_path}")
    print()

print("Patch inference completed.")