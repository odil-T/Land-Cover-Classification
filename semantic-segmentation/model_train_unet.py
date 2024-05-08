"""
Trains and saves a U-Net model on OpenEarthMap Dataset.
"""

import os
import pickle
import re
import cv2
import torch.cuda
import numpy as np
import datetime
import torch.nn.functional as F
import dotenv
from PIL import Image
from tqdm import tqdm
from unet_torch import UNet
from torch import nn
from torchmetrics import JaccardIndex
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


dotenv.load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

# Specify input and output paths
root_data_dir = "data/OpenEarthMap/OpenEarthMap_wo_xBD"
train_txt_file = "train_wo_xBD.txt"
val_txt_file = "val_wo_xBD.txt"
outputs_save_dir = f"models/unet_sem_seg_{current_datetime}"

num_classes = int(os.getenv("NUM_CLASSES"))
height = int(os.getenv("TARGET_HEIGHT"))
width = int(os.getenv("TARGET_WIDTH"))

epochs = 50
batch_size = int(os.getenv("BATCH_SIZE"))
learning_rate = 1e-3


class DiceLoss(nn.Module):
    """
    Dice loss for multiclass segmentation.
    """

    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Tensor of shape (batch_size, num_classes, height, width).
            Represents the predicted probabilities or logits.
            targets (torch.Tensor): Tensor of shape (batch_size, height, width). Represents the ground truth labels.
            Should contain class indices in the range [0, num_classes - 1].

        Returns:
            torch.Tensor: Average dice loss.
        """

        num_classes = inputs.shape[1]

        if inputs.dtype == torch.float32:
            inputs = torch.softmax(inputs, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (batch_size, num_classes, height, width)

        dice_losses = []
        for c in range(num_classes):
            input_class = inputs[:, c, :, :]
            target_class = targets_one_hot[:, c, :, :]

            intersection = torch.sum(input_class * target_class, dim=(1, 2))
            union = torch.sum(input_class + target_class, dim=(1, 2))

            dice_coeff = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss = 1 - dice_coeff
            dice_losses.append(dice_loss)

        avg_dice_loss = torch.mean(torch.stack(dice_losses))

        return avg_dice_loss


class EarlyStopping():
    """
    Breaks model optimization loop if no specified amount of improvement is made after a specified number of epochs.

    Attributes:
        patience (int): The number of epochs allowed where no improvement is made in the validation loss.
        min_delta (float): The minimum decrease in validation loss required after an epoch of training to be considered as an improvement.
        patience_counter (int): The count of epochs where improvement is not made.
        best_val_loss (float): The current lowest validation loss obtained.
        early_stop (bool): Indicates whether the optimization loop should be stopped.
    """

    def __init__(self, patience=3, min_delta=0.0):
        """
        Args:
              patience (int): The number of epochs allowed where improvement is not made.
              min_delta (float): The minimum decrease in validation loss required after an epoch of training to be considered as an improvement.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_val_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stop = True


class OpenEarthMapDataset(Dataset):
    """
    PyTorch Dataset implementation of the Open Earth Map Dataset.

    Attributes:
        root_data_dir (str): Root directory path of Open Earth Map Dataset from which to load images, masks, and txt files.
        filenames (list): List of file names of images (and masks) that must be used for training. Both images and masks
        have the same names. They are stored in different directories.
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


def train_loop(dataloader, model, loss_function, optimizer):
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

        pred = model(batch_X)
        loss = loss_function(pred, batch_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        miou_metric.update(pred, batch_y)

    avg_train_loss = train_loss / len(dataloader)
    train_mean_iou = miou_metric.compute()
    print(f"Training Loss: {avg_train_loss:.4f}, Training Mean IoU: {train_mean_iou:.4f}")

    # Logging optimization history
    train_loss_history.append(avg_train_loss)
    train_miou_history.append(train_mean_iou)


def val_loop(dataloader, model, loss_function):
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

            pred = model(batch_X)
            loss = loss_function(pred, batch_y)

            val_loss += loss.item()
            miou_metric.update(pred, batch_y)

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
train_dataset = OpenEarthMapDataset(root_data_dir, train_txt_file, (height, width))
val_dataset = OpenEarthMapDataset(root_data_dir, val_txt_file, (height, width))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Model Preparation
model = UNet(3, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()
miou_metric = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
early_stopping = EarlyStopping(5, 0.01)

# Logging History
train_loss_history = []
val_loss_history = []
train_miou_history = []
val_miou_history = []

# Optimization Loop
for epoch in range(epochs):
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    print(f"Epoch {epoch + 1}/{epochs}")

    train_loop(train_dataloader, model, loss_function, optimizer)
    avg_val_loss = val_loop(val_dataloader, model, loss_function)

    # Save a model checkpoint every 5 epochs
    if epoch % 5 == 0 and epoch != 0:
        save_checkpoint(model, optimizer, epoch,
                        os.path.join(outputs_save_dir, f"unet_sem_seg_checkpoint_epoch{epoch}.pt"))

    early_stopping(avg_val_loss)

history = {
    "train_loss": train_loss_history,
    "val_loss": val_loss_history,
    "train_miou": train_miou_history,
    "val_miou": val_miou_history
}

# Save Model & Train History
torch.save(model, os.path.join(outputs_save_dir, "unet_sem_seg.pth"))
with open(os.path.join(outputs_save_dir, "history.pkl"), "wb") as f:
    pickle.dump(history, f)
