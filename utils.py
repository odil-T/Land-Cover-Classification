"""
Contains utility functions.
"""

import cv2
import numpy as np


def resize_and_pad(image, target_size, is_mask):
    """
    Resizes an image or mask to target size. The image/mask is resized along its longest dimension while maintaining
    the aspect ratio. If necessary, padding is applied to the regions along the shorter dimension to match the given
    target size. Black color is used for padding the image. Class ID 0 is used for padding the mask.

    Args:
        image (numpy.ndarray): The image or mask that needs to be resized.
        target_size (int): The target size for which the image or mask needs to be resized to.
        is_mask (bool): Specifies whether the given image is an RGB image or a one-channel mask.

    Returns:
        numpy.ndarray: The resized image.
    """

    original_aspect = image.shape[1] / image.shape[0]
    target_aspect = 1

    if original_aspect > target_aspect:
        new_width = target_size
        new_height = int(new_width / original_aspect)
    else:
        new_height = target_size
        new_width = int(new_height * original_aspect)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LANCZOS4
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    pad_top = (target_size - new_height) // 2
    pad_bottom = target_size - new_height - pad_top
    pad_left = (target_size - new_width) // 2
    pad_right = target_size - new_width - pad_left

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


def crop_center(image, crop_size):
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


def random_color(blacklisted_colors):
    """
    Generates a random color different from the blacklisted colors.

    Args:
        blacklisted_colors (list): A list of colors that should not be generated.

    Returns:
        list: The randomly generated color.
    """
    color = None
    while color is None or color in blacklisted_colors:
        color = np.random.randint(0, 255, size=3).tolist()
    return color