from sklearn.metrics import confusion_matrix
import os
import pickle
import re
import cv2
import torch.cuda
import numpy as np
import datetime
import dotenv
import evaluate
from PIL import Image
from torch import nn
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation


dotenv.load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

# Specify input and output paths
root_data_dir = "data/OpenEarthMap/OpenEarthMap_wo_xBD"
train_txt_file = "train.txt"
val_txt_file = "val.txt"
outputs_save_dir = f"models/segformer_sem_seg_{current_datetime}"
os.makedirs(outputs_save_dir)

num_classes = int(os.getenv("NUM_CLASSES"))
height = int(os.getenv("TARGET_HEIGHT"))
width = int(os.getenv("TARGET_WIDTH"))

epochs = 100
batch_size = int(os.getenv("BATCH_SIZE"))
learning_rate = 6e-5


def calculate_metrics(preds, labels):
    # Compute mIOU for each class
    iou_per_class = []
    for cls in range(num_classes):
        intersection = torch.logical_and(labels == cls, preds == cls).sum().float()
        union = torch.logical_or(labels == cls, preds == cls).sum().float()
        iou = (intersection / (union + 1e-8)).item()
        iou_per_class.append(iou)
    mIOU = sum(iou_per_class) / num_classes

    # Compute overall accuracy
    accuracy = (preds == labels).float().mean().item()

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

    def __init__(self, root_data_dir, filenames_file, target_size):
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
        location_dir = str(re.search(r'^(.*)_(?=\d)', filename).group(1))  # for e.g. "aachen"

        image_path = os.path.join(self.root_data_dir, location_dir, "images", filename)
        mask_path = os.path.join(self.root_data_dir, location_dir, "labels", filename)

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image_width, image_height = image.size

        if image_width >= self.target_size[1] and image_height >= self.target_size[0]:
            image = self.crop_center(image, self.target_size[0])
            mask = self.crop_center(mask, self.target_size[0])
            mask = np.array(mask)

        else:
            image = np.array(image)
            image = self.resize_and_pad(image, self.target_size, False)

            mask = np.array(mask)
            mask = self.resize_and_pad(mask, self.target_size, True)

        image = ToTensor()(image)
        mask = torch.from_numpy(mask).type(torch.LongTensor)

        return image, mask  # (3, height, width), (height, width)


# Data Preparation
val_dataset = OpenEarthMapDataset(root_data_dir, val_txt_file, (height, width))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Assuming your validation dataset is stored in a DataLoader called val_loader

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2",
                                                         num_labels=num_classes).to(device)
checkpoint = torch.load("models/segformer_sem_seg_2024-05-16--14-40-45/segformer_sem_seg_checkpoint_epoch40.pt")
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

        pred_logits = model(images).logits  # output: torch.Tensor (1, 9, 250, 250)
        upsampled_logits = nn.functional.interpolate(pred_logits,
                                                     size=(height, width), mode="bilinear",
                                                     align_corners=False)  # output: torch.Tensor (1, 9, 1000, 1000)
        pred = upsampled_logits.argmax(dim=1)
        pred = pred.cpu()

        iou, acc = calculate_metrics(pred, labels)
        val_miou += iou
        val_accuracy += acc
        total_batches += 1

val_miou /= total_batches
val_accuracy /= total_batches


print("epoch40")
print(val_miou)
print(val_accuracy)