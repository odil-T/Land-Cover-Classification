from PIL import Image


def crop_image(image_path, crop_size):

    image = Image.open(image_path)
    width, height = image.size
    crops = []

    if width % crop_size == 0 and height % crop_size == 0:
        for i in range(0, width, crop_size):
            for j in range(0, height, crop_size):
                box = (i, j, i + crop_size, j + crop_size)
                crop = image.crop(box)
                crops.append(crop)

    else:
        new_width = width - (width % crop_size)
        new_height = height - (height % crop_size)

        for i in range(0, new_width, crop_size):

            for j in range(0, new_height, crop_size):
                box = (i, j, i + crop_size, j + crop_size)
                crop = image.crop(box)
                crops.append(crop)

                # Extra overlapping crop after last iteration
                if j == new_height - crop_size:
                    box = (i, height - crop_size, i + crop_size, height)
                    crop = image.crop(box)
                    crops.append(crop)

                # Extra overlapping crop after last iteration
                if i == new_width - crop_size:
                    box = (width - crop_size, j, width, j + crop_size)
                    crop = image.crop(box)
                    crops.append(crop)

    return crops


crops = crop_image("lmao.jpg", 512)

for crop in crops:
    crop.show()