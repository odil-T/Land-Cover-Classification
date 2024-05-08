import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV is commonly used for image operations


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


# Create an example image (RGB)
image = cv2.imread("data/test_samples/test1.png")
image = resize_and_pad(image, (1024, 1024), False)
mask = cv2.imread("data/test_samples/test1_mask.png")

alpha = 1  # 0 = only image, 1 = only mask

# Blend the image and mask
blended = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

# Display the result
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.title("Image with Mask Overlay")
plt.axis("off")
plt.show()


