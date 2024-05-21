from sklearn.metrics import confusion_matrix
import os
import pickle
import re
import cv2
import torch.cuda
import numpy as np
import datetime
import dotenv
from PIL import Image
from torch import nn
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from unet_torch import *


dotenv.load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")


num_classes = int(os.getenv("NUM_CLASSES"))
height = 1024
width = 1024

batch_size = int(os.getenv("BATCH_SIZE"))


def calculate_metrics(preds, labels, num_classes, ignore_index=None):
    if ignore_index is None:
        ignore_index = []

    # Compute mIOU for each class
    iou_per_class = []
    for cls in range(num_classes):
        if cls in ignore_index:
            continue

        intersection = torch.logical_and(labels == cls, preds == cls).sum().float()
        union = torch.logical_or(labels == cls, preds == cls).sum().float()
        iou = (intersection / (union + 1e-8)).item()
        iou_per_class.append(iou)

    mIOU = sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0.0

    # Compute overall accuracy
    mask = torch.ones_like(labels, dtype=torch.bool)
    for cls in ignore_index:
        mask &= (labels != cls)

    valid_preds = preds[mask]
    valid_labels = labels[mask]

    accuracy = (valid_preds == valid_labels).float().mean().item() if valid_labels.numel() > 0 else 0.0

    return mIOU, accuracy


class OpenEarthMapDataset(Dataset):
    """
    PyTorch Dataset implementation of the Open Earth Map Dataset.

    Attributes:
        root_data_dir (str): Root directory path of Open Earth Map Dataset from which to load images, masks, and txt files.
        filenames (list): List of file names of images (and masks) that must be used for training. Both images and masks
        have the same names. They are stored in different directories.
        target_size (tuple): Target size of the image and mask to be resized to for model training.
    """

    def __init__(self, rural_urban_dir_path, target_size):
        """
        Args:
            root_data_dir (str): Root directory path of Open Earth Map Dataset.
            filenames_file (str): Name of txt file that stores the file names of images and masks to be used for training.
            target_size (tuple): Target size of the image and mask to be resized to for model training.
        """

        self.ru_dirpath = rural_urban_dir_path
        self.filenames = os.listdir(os.path.join(rural_urban_dir_path, "images_png"))
        self.target_size = target_size

    def crop_center(self, image, crop_size):
        """
        Crops the image from its center. The crop is a square that has a length of specified size.

        Args:
            image (PIL.Image): The pillow image to crop.
            crop_size (int): The length of the crop square.

        Returns:
            PIL.Image: The cropped pillow image.
        """

        width, height = image.size
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2
        return image.crop((left, top, right, bottom))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        """
        Loads the next image and mask to be used for training. The image is resized and normalized. The mask is resized.

        Args:
            item (int): Index of the image and mask to load.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): Tensor of image with shape (3, *self.target_size).
                - mask (torch.Tensor): Tensor of mask with shape self.target_size.
        """

        filename = self.filenames[item]  # for e.g. "aachen_1.tif"



        image_path = os.path.join(self.ru_dirpath, "images_png", filename)
        mask_path = os.path.join(self.ru_dirpath, "masks_png", filename)

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = self.crop_center(image, self.target_size[0])
        mask = self.crop_center(mask, self.target_size[0])

        mask = np.array(mask)

        image = ToTensor()(image)
        mask = torch.from_numpy(mask).type(torch.LongTensor)

        return image, mask  # (3, height, width), (height, width)


# Data Preparation
val_dataset = OpenEarthMapDataset("data/LoveDA/Val/Rural", (height, width))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



model = UNet(3, num_classes).to(device)
checkpoint = torch.load("models/unet_sem_seg_2024-05-07--05-45-34/unet_sem_seg_checkpoint_epoch40.pt")
model.load_state_dict(checkpoint["model_state_dict"])




model.eval()
val_miou = 0.0
val_accuracy = 0.0
total_batches = 0

with torch.no_grad():
    for images, labels in val_dataloader:
        images = images.to(device)
        labels = labels

        # Forward pass
        outputs = model(images)

        outputs = nn.Softmax(dim=1)(outputs)
        outputs = outputs.argmax(dim=1)
        pred = outputs.cpu()

        pred[pred == 0] = 11
        pred[pred == 1] = 55
        pred[pred == 2] = 0
        pred[pred == 3] = 0
        pred[pred == 4] = 3
        pred[pred == 5] = 66
        pred[pred == 6] = 4
        pred[pred == 8] = 2

        pred[pred == 11] = 1
        pred[pred == 55] = 5
        pred[pred == 66] = 6


        iou, acc = calculate_metrics(outputs, labels, num_classes, [0])
        val_miou += iou
        val_accuracy += acc
        total_batches += 1

val_miou /= total_batches
val_accuracy /= total_batches


print("rural")
print(val_miou)
print(val_accuracy)