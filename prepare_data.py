import os
import cv2
import patchify
import numpy as np
import tensorflow as tf
from unet_tf import build_unet
from PIL import Image
from sklearn.model_selection import train_test_split


patch_size = 128

# Class mask colors in order
color_mapping = [[60, 16, 152],
                 [132, 41, 246],
                 [110, 193, 228],
                 [254, 221, 58],
                 [226, 169, 41],
                 [155, 155, 155]]
n_classes = len(color_mapping)

def preprocess_data():
    """Loads images and masks, resizes, normalizes, patchifies images and masks, converts the masks RGB array to an
    integer class code array and outputs datasets as arrays."""

    images_dataset = []
    masks_dataset = []

    for root_path, dir_list, file_list in os.walk("data"):
        if "images" in root_path:
            for image_name in file_list:
                image_path = os.path.join(root_path, image_name)
                image = cv2.imread(image_path)

                # Cropping with pillow
                height = (image.shape[0] // patch_size) * patch_size
                width = (image.shape[1] // patch_size) * patch_size
                image = Image.fromarray(image)
                image = image.crop((0, 0, width, height))
                image = np.array(image)

                # Normalization
                image = image.astype(float) / 255.

                # Dividing image into patches
                patched_images = patchify.patchify(image, (patch_size, patch_size, 3), step=patch_size)

                for i in range(patched_images.shape[0]):
                    for j in range(patched_images.shape[1]):
                        patch = patched_images[i, j, 0]
                        images_dataset.append(patch)

        elif "masks" in root_path:
            for mask_name in file_list:
                mask_path = os.path.join(root_path, mask_name)
                mask = cv2.imread(mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                # Cropping with pillow
                height = (mask.shape[0] // patch_size) * patch_size
                width = (mask.shape[1] // patch_size) * patch_size
                mask = Image.fromarray(mask)
                mask = mask.crop((0, 0, width, height))
                mask = np.array(mask)

                # Dividing image into patches
                patched_masks = patchify.patchify(mask, (patch_size, patch_size, 3), step=patch_size)

                for i in range(patched_masks.shape[0]):
                    for j in range(patched_masks.shape[1]):
                        patch = patched_masks[i, j, 0]
                        masks_dataset.append(patch)

    images_dataset = np.array(images_dataset)  # (5535, 128, 128, 3)
    masks_dataset = np.array(masks_dataset)  # (5535, 128, 128, 3)

    # Converting mask RGBs to integer class codes
    for i in range(len(color_mapping)):
        mask = np.all(masks_dataset == color_mapping[i], axis=-1)
        masks_dataset[mask] = i

    masks_dataset = masks_dataset[:, :, :, 0]  # (5535, 128, 128)

    # (5535, 128, 128, 3), (5535, 128, 128)
    return images_dataset, masks_dataset

images_dataset, masks_dataset = preprocess_data()

X_train, X_valid, y_train, y_valid = train_test_split(images_dataset, masks_dataset, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(X_test.shape)
print(y_test.shape)
