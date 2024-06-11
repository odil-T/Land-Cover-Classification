"""
Perform panoptic segmentation on a satellite image by merging inferred semantic and instance masks.
You may specify the image to infer, the SegFormer semantic model, and the YOLO instance model.
"""

import os
import torch
from dotenv import load_dotenv
from utils import *
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import ToTensor
from transformers import SegformerForSemanticSegmentation
from ultralytics import YOLO


load_dotenv()

num_classes = int(os.getenv("NUM_CLASSES"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify path of image to infer
image_path = "instance-segmentation/img2.jpg"

# Load SegFormer model
segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2",
                                                         num_labels=num_classes).to(device)
checkpoint = torch.load("best_models/segformer_sem_seg_checkpoint_epoch35.pt")
segformer.load_state_dict(checkpoint["model_state_dict"])

# Load YOLO model
yolo = YOLO("best_models/yolov8n-seg.pt")


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
    Crops the image center.

    Args:
        image_path:
        target_size:

    Returns:
        numpy.ndarray:
    """

    image = Image.open(image_path)
    image_width, image_height = image.size

    if image_width >= target_size and image_height >= target_size:
        image_original = crop_center(image, target_size)
        image_original = np.array(image_original)
    else:
        image = np.array(image)
        image_original = resize_and_pad(image, (target_size, target_size), False)

    return image_original


def infer_segformer(image_path, model, target_size, colormap):
    """

    Args:
        image (numpy.ndarray):
        model:
        target_size:
        colormap:

    Returns:

    """

    colormap = colormap[:num_classes]

    image = preprocess_image(image_path, target_size)
    image = ToTensor()(image).to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        pred_logits = model(image).logits  # output: torch.Tensor (1, 9, 250, 250)
        upsampled_logits = nn.functional.interpolate(pred_logits,
                                                     size=(target_size, target_size), mode="bilinear",
                                                     align_corners=False)  # output: torch.Tensor (1, 9, 1000, 1000)
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

    Args:
        image (numpy.ndarray): An RGB image.
        model:
        blacklisted_colors:

    Returns:
        numpy.ndarray:
    """

    image = preprocess_image(image_path, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(image)

    for result in results:
        predicted_image = np.zeros(image.shape, dtype=int)

        # Drawing predicted instance masks
        polygons = result.masks.cpu().xy

        for polygon in polygons:
            polygon = np.array(polygon).astype(np.int32)

            color = random_color(blacklisted_colors)
            blacklisted_colors.append(color)

            cv2.fillPoly(predicted_image, [polygon], color)

        return predicted_image


def display(image_path):

    original_image = np.array(Image.open(image_path))

    semantic_mask = infer_segformer(image_path, segformer, target_size, colormap)
    instance_mask = infer_yolo(image_path, yolo, target_size, colormap)

    print('SEMANTIC', semantic_mask, semantic_mask.shape)

    print('INSTANCE', instance_mask, instance_mask.shape)

    # Merging the masks
    black_mask = np.all(instance_mask == [0, 0, 0], axis=-1)
    overlay_mask = ~black_mask
    overlay_mask = overlay_mask.astype(np.uint8)
    overlay_mask = np.expand_dims(overlay_mask, axis=-1)
    merged_mask = semantic_mask * (1 - overlay_mask) + instance_mask * overlay_mask

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(original_image)
    axs[0].axis('off')
    axs[0].set_title("Original Image")

    axs[1].imshow(merged_mask)
    axs[1].axis('off')
    axs[1].set_title("Predicted Merged Mask")

    plt.show()


display(image_path)