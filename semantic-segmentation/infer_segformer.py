"""
Displays original images from the validation set along with their ground truth masks and predicted masks for comparison.
The predicted masks are produced by a pretrained SegFormer model.
The samples are taken randomly.
You may specify the number of samples to display.
"""

import os
import re
import dotenv
import random
import numpy as np
from utils import resize_and_pad, crop_center
from unet_torch import *
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from transformers import SegformerForSemanticSegmentation


dotenv.load_dotenv()

# Specify number of samples to display from the validation set
num_samples = 5

colormap = [
    (0, 0, 0),        # Black           --- Background
    (128, 0, 0),      # Maroon          --- Bareland
    (0, 255, 36),     # Light Green     --- Rangeland
    (148, 148, 148),  # Grey            --- Developed space
    (255, 255, 255),  # White           --- Road
    (34, 97, 38),     # Dark Green      --- Tree
    (0, 69, 255),     # Blue            --- Water
    (75, 181, 73),    # Green           --- Agriculture land
    (222, 31, 7),     # Red             ---	Building
]
num_classes = int(os.getenv("NUM_CLASSES"))
device = "cuda" if torch.cuda.is_available() else "cpu"


def display_results(n, model):
    """
    Displays random N number of images, ground truth masks, and predicted masks from the validation set of the
    OpenEarthMap dataset.

    Args:
        n (int): The number of random samples from the validation set to display.
        model: PyTorch model to use for semantic segmentation.
    """

    height = 1000
    width = 1000

    root_data_dir = "data/OpenEarthMap/OpenEarthMap_wo_xBD"
    filenames_file = "val.txt"

    with open(os.path.join(root_data_dir, filenames_file), "r") as f:
        filenames = [re.sub(r'\n+$', '', line) for line in f.readlines()]  # ["aachen_1.tif", "aachen_10.tif", ...]

    random_n_filenames = random.sample(filenames, n)

    for filename in random_n_filenames:
        location_dir = str(re.search(r'^(.*)_(?=\d)', filename).group(1))  # for e.g. "aachen"

        image_path = os.path.join(root_data_dir, location_dir, "images", filename)
        mask_path = os.path.join(root_data_dir, location_dir, "labels", filename)

        image = Image.open(image_path)
        true_mask = Image.open(mask_path)

        image_width, image_height = image.size

        if image_width >= width and image_height >= height:
            image_original = crop_center(image, height)
            true_mask = crop_center(true_mask, height)
            true_mask = np.array(true_mask)

        else:
            image = np.array(image)
            image_original = resize_and_pad(image, (height, width), False)

            true_mask = np.array(true_mask)
            true_mask = resize_and_pad(true_mask, (height, width), True)


        image = ToTensor()(image_original).to(device)
        image = image.unsqueeze(0)

        with torch.no_grad():
            pred_logits = model(image).logits  # output: torch.Tensor (1, 9, 250, 250)
            upsampled_logits = nn.functional.interpolate(pred_logits,
                                                         size=(height, width), mode="bilinear",
                                                         align_corners=False)  # output: torch.Tensor (1, 9, 1000, 1000)
            pred = upsampled_logits.argmax(dim=1)
            pred = pred.squeeze(0)
            pred = pred.cpu()

        # True mask
        true_mask_output = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(colormap):
            true_mask_output[true_mask == class_id] = color

        # Predicted mask
        pred_mask_output = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(colormap):
            pred_mask_output[pred == class_id] = color


        # Display images
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))

        axs[0].imshow(image_original, cmap='gray')
        axs[0].axis('off')
        axs[0].set_title("Original Image")

        axs[1].imshow(true_mask_output)
        axs[1].axis('off')
        axs[1].set_title("Ground Truth Mask")

        axs[2].imshow(pred_mask_output)
        axs[2].axis('off')
        axs[2].set_title("Predicted Mask")

        plt.show()


# Load pretrained model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=num_classes).to(device)

display_results(num_samples, model)
