import os
import pickle
import re
import cv2
import torch.cuda
import numpy as np
import datetime
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from unet_torch import *
from torch import nn
from torchmetrics import JaccardIndex
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify input and output paths
root_data_dir = "../data/OpenEarthMap/OpenEarthMap_wo_xBD"
test_txt_file = "test_wo_xBD.txt"

num_classes = 9
height, width = 1024, 1024
batch_size = 4


class OpenEarthMapDataset(Dataset):
    """
    PyTorch Dataset implementation of the Open Earth Map Dataset.

    Attributes:
        root_data_dir (str): Root directory path of Open Earth Map Dataset from which to load images, masks, and txt files.
        filenames (list): List of file names of images (and masks) that must be used for training. Both images and masks
        have the same names. THey are stored in different directories.
        target_size (tuple): Target size of the image and mask to be resized to for model training.
    """

    def __init__(self, root_data_dir, filenames_file, target_size=(1024, 1024)):
        """
        Args:
            root_data_dir (str): Root directory path of Open Earth Map Dataset.
            filenames_file (str): Name of txt file that stores the file names of images and masks to be used for training.
            target_size (tuple): Target size of the image and mask to be resized to for model training.
        """

        self.root_data_dir = root_data_dir
        with open(os.path.join(root_data_dir, filenames_file), "r") as f:
            self.filenames = [re.sub(r'\n+$', '', line) for line in f.readlines()]  #  ["aachen_1.tif", "aachen_10.tif", ...]
        self.target_size = target_size

    def resize_and_pad(self, image, target_size, is_mask):
        """
        Resizes an image or mask to target size. The image/mask is resized along its longest dimension while maintaining
        the aspect ratio. If necessary, padding is applied to the regions along the shorter dimension to match the given
        target size. Black color is used for padding the image. Class ID 0 is used for padding the mask.

        Args:
            image (numpy.ndarray): The image or mask that needs to be resized.
            target_size (tuple): The target size for which the image or mask needs to be resized to.
            is_mask (bool): Specifies whether the given image is an RGB image or a one-channel mask.

        Returns:
            numpy.ndarray: The resized image.
        """

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
        location_dir = str(re.search(r'^(.*)_(?=\d)', filename).group(1))  # for e.g. "aachen"

        image_path = os.path.join(self.root_data_dir, location_dir, "images", filename)
        mask_path = os.path.join(self.root_data_dir, location_dir, "labels", filename)

        image = Image.open(image_path)
        image = np.array(image)
        image = self.resize_and_pad(image, self.target_size, False)
        image = ToTensor()(image)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = self.resize_and_pad(mask, self.target_size, True)
        mask = torch.from_numpy(mask).type(torch.LongTensor)

        return image, mask  # (3, 1024, 1024), (1024, 1024)


# Data Preparation
test_dataset = OpenEarthMapDataset(root_data_dir, test_txt_file, (height, width))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model_path = "../models/unet_sem_seg_2024-05-05--13-07-27/unet_sem_seg.pth"
model = torch.load(model_path)

X, y = next(iter(test_dataloader))

print(X.shape)
print(y.shape)