import cv2
import numpy as np
import tensorflow as tf
import patchify

patch_size = 128
infer_image_path = "sample.png"


def resize_image(image, patch_size):
    """Resizes a to-be-inferred image to the highest multiple of the patch size."""

    width_and_height = [(image.shape[i] // patch_size) * patch_size for i in range(1, -1, -1)]
    image = cv2.resize(image, width_and_height)

    return image


model = tf.keras.models.Model()

# Image to be inferred
image = cv2.imread(infer_image_path)
image = resize_image(image, patch_size)

# Placeholder array for output image
output_image = np.zeros((image.shape[:2]))

patched_images = patchify.patchify(image, (patch_size, patch_size, 3), step=patch_size)

for i in range(patched_images.shape[0]):
    height_pixel_position = i * patch_size

    for j in range(patched_images.shape[1]):
        patch = patched_images[i, j, 0]  # (128, 128, 3)
        probability_patch = model.predict(patch)  # (patch_size, patch_size, n_classes)
        mask_patch = np.argmax(probability_patch, axis=-1)  # class code mask | (128, 128)

        width_pixel_position = j * patch_size

        # Replaces the placeholder array with the patches
        output_image[height_pixel_position:height_pixel_position+patch_size, \
        width_pixel_position:width_pixel_position+patch_size] = mask_patch

# output_image contains the mask for the classes as integer codes
# apply this mask to the original image
# do not apply for background
