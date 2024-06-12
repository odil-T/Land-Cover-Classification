"""
Trains and saves a SegFormer model on OpenEarthMap Dataset for semantic segmentation.
The entire OpenEarthMap Dataset is included along with xBD RGB images.
Image centers are cropped if they are larger than target size. If less, they are resized and padded.
"""

import os
import pickle
import re
import torch.cuda
import numpy as np
import datetime
from utils import resize_and_pad, crop_center
from PIL import Image
from torch import nn
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation


device = "cuda" if torch.cuda.is_available() else "cpu"
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

# Specify input and output paths
root_data_dir = "data/OpenEarthMap/OpenEarthMap_wo_xBD"
train_txt_file = "train.txt"
val_txt_file = "val.txt"
outputs_save_dir = f"models/segformer_sem_seg_{current_datetime}"
os.makedirs(outputs_save_dir)

num_classes = 9
target_size = 650

epochs = 50
batch_size = 2
learning_rate = 6e-5


class OpenEarthMapDataset(Dataset):
    """
    PyTorch Dataset implementation of the Open Earth Map Dataset.

    Attributes:
        root_data_dir (str): Root directory path of Open Earth Map Dataset from which to load images, masks, and txt files.
        filenames (list): List of file names of images (and masks) that must be used for training. Both images and masks
        have the same names. They are stored in different directories.
        target_size (int): Height and width of the image and mask to be resized to for model training.
    """

    def __init__(self, root_data_dir, filenames_file, target_size):
        """
        Args:
            root_data_dir (str): Root directory path of Open Earth Map Dataset.
            filenames_file (str): Name of txt file that stores the file names of images and masks to be used for training.
            target_size (int): Height and width of the image and mask to be resized to for model training.
        """

        self.root_data_dir = root_data_dir
        with open(os.path.join(root_data_dir, filenames_file), "r") as f:
            self.filenames = [re.sub(r'\n+$', '', line) for line in f.readlines()]  #  ["aachen_1.tif", "aachen_10.tif", ...]
        self.target_size = target_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        """
        Loads the next image and mask to be used for training. The image is resized and normalized. The mask is resized.

        Args:
            item (int): Index of the image and mask to load.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): Tensor of image with shape (3, self.target_size, self.target_size).
                - mask (torch.Tensor): Tensor of mask with shape (self.target_size, self.target_size).
        """

        filename = self.filenames[item]  # for e.g. "aachen_1.tif"
        location_dir = str(re.search(r'^(.*)_(?=\d)', filename).group(1))  # for e.g. "aachen"

        image_path = os.path.join(self.root_data_dir, location_dir, "images", filename)
        mask_path = os.path.join(self.root_data_dir, location_dir, "labels", filename)

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image_width, image_height = image.size

        if image_width >= self.target_size and image_height >= self.target_size:
            image = crop_center(image, self.target_size)
            mask = crop_center(mask, self.target_size)
            mask = np.array(mask)

        else:
            image = np.array(image)
            image = resize_and_pad(image, self.target_size, False)

            mask = np.array(mask)
            mask = resize_and_pad(mask, self.target_size, True)

        image = ToTensor()(image)
        mask = torch.from_numpy(mask).type(torch.LongTensor)

        return image, mask  # (3, height, width), (height, width)


def train_loop(dataloader, model, optimizer):
    """
    The train loop portion of the optimization loop. This function is called at every epoch.

    Args:
        dataloader: PyTorch Dataloader that loads the train set in batches.
        model: PyTorch model.
        loss_function: PyTorch loss function.
        optimizer: PyTorch optimizer.
    """

    model.train()
    train_loss = 0.0
    miou_metric.reset()

    for batch_X, batch_y in tqdm(dataloader):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(pixel_values=batch_X, labels=batch_y)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        with torch.no_grad():
            upsampled_logits = nn.functional.interpolate(logits, size=batch_y.shape[-2:], mode="bilinear",
                                                         align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

        miou_metric.update(predicted, batch_y)

    avg_train_loss = train_loss / len(dataloader)
    train_mean_iou = miou_metric.compute()
    print(f"Training Loss: {avg_train_loss:.4f}, Training Mean IoU: {train_mean_iou:.4f}")

    # Logging optimization history
    train_loss_history.append(avg_train_loss)
    train_miou_history.append(train_mean_iou)


def val_loop(dataloader, model):
    """
    The validation loop portion of the optimization loop. This function is called at every epoch.

    Args:
        dataloader: PyTorch Dataloader that loads the validation set in batches.
        model: PyTorch model.
        loss_function: PyTorch loss function.

    Returns:
        float: Average validation loss.
    """

    model.eval()
    miou_metric.reset()
    val_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in tqdm(dataloader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(pixel_values=batch_X, labels=batch_y)
            loss, logits = outputs.loss, outputs.logits

            val_loss += loss.item()

            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(logits, size=batch_y.shape[-2:], mode="bilinear",
                                                             align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

            miou_metric.update(predicted, batch_y)

    avg_val_loss = val_loss / len(dataloader)
    val_mean_iou = miou_metric.compute()
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Mean IoU: {val_mean_iou:.4f}\n")

    # Logging optimization history
    val_loss_history.append(avg_val_loss)
    val_miou_history.append(val_mean_iou)

    return avg_val_loss


def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves a model checkpoint.

    Args:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        epoch (int): The current epoch number.
        path (str): Checkpoint save path.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


# Data Preparation
train_dataset = OpenEarthMapDataset(root_data_dir, train_txt_file, target_size)
val_dataset = OpenEarthMapDataset(root_data_dir, val_txt_file, target_size)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Model Preparation
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2",
                                                         num_labels=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
miou_metric = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)

# Logging History
train_loss_history = []
val_loss_history = []
train_miou_history = []
val_miou_history = []

# Optimization Loop
for epoch in range(epochs):

    print(f"Epoch {epoch + 1}/{epochs}")

    train_loop(train_dataloader, model, optimizer)
    avg_val_loss = val_loop(val_dataloader, model,)

    # Save a model checkpoint every 5 epochs
    if epoch % 5 == 0 and epoch != 0:
        save_checkpoint(model, optimizer, epoch,
                        os.path.join(outputs_save_dir, f"segformer_sem_seg_checkpoint_epoch{epoch}.pt"))


history = {
    "train_loss": train_loss_history,
    "val_loss": val_loss_history,
    "train_miou": train_miou_history,
    "val_miou": val_miou_history
}

# Save Model & Train History
torch.save(model, os.path.join(outputs_save_dir, "segformer_sem_seg.pth"))
with open(os.path.join(outputs_save_dir, "history.pkl"), "wb") as f:
    pickle.dump(history, f)
