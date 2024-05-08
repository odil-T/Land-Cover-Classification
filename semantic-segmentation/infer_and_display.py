import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from unet_torch import *
from PIL import Image
import os
import dotenv


dotenv.load_dotenv()
colormap = [
    (0, 0, 0),        # Black for the first class
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (0, 255, 255),    # Cyan
    (255, 192, 203)   # Pink
]


def resize_and_pad(image, target_size, is_mask):
    original_aspect = image.shape[1] / image.shape[0]
    target_aspect = target_size[0] / target_size[1]

    if original_aspect > target_aspect:
        new_width = target_size[0]
        new_height = int(new_width / original_aspect)
    else:
        new_height = target_size[1]
        new_width = int(new_height * original_aspect)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LANCZOS4
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    pad_top = (target_size[1] - new_height) // 2
    pad_bottom = target_size[1] - new_height - pad_top
    pad_left = (target_size[0] - new_width) // 2
    pad_right = target_size[0] - new_width - pad_left

    padding_value = [0, 0, 0] if not is_mask else 0
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=padding_value,
    )

    return padded_image


def display_results(model, target_size, image_path):
    """
    Displays an image, its ground truth mask, and its prediction mask based on the provided image path.

    Args:
        model:
        target_size:
        image_path:
    """

    image = Image.open(image_path)
    image = np.array(image)
    image_original = resize_and_pad(image, target_size, False)
    image = ToTensor()(image_original).to(device)
    image = image.unsqueeze(0)

    true_mask = Image.open(image_path.replace("images", "labels"))
    true_mask = np.array(true_mask)

    with torch.no_grad():
        pred = model(image)  # torch.Tensor (1, 9, 1024, 1024)

    pred = nn.Softmax(dim=1)(pred)
    pred = pred.cpu()
    pred = np.argmax(pred, axis=1)
    pred = pred.squeeze(0)

    # true mask
    true_mask_output = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colormap):
        true_mask_output[true_mask == class_id] = color

    # predicted mask
    pred_mask_output = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colormap):
        pred_mask_output[pred == class_id] = color


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


height = int(os.getenv("TARGET_HEIGHT"))
width = int(os.getenv("TARGET_WIDTH"))
num_classes = int(os.getenv("NUM_CLASSES"))

device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "data/OpenEarthMap/OpenEarthMap_wo_xBD/viru/images/viru_47.tif"

# Load Model
# model = torch.load("models/unet_sem_seg_2024-05-05--13-07-27/unet_sem_seg.pth")  # old model
model = UNet(3, num_classes).to(device)
checkpoint = torch.load("models/unet_sem_seg_2024-05-07--05-45-34/unet_sem_seg_checkpoint_epoch40.pt")
model.load_state_dict(checkpoint["model_state_dict"])  # new model


display_results(model, (height, width), image_path)
