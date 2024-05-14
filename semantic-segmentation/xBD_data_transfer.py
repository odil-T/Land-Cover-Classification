"""
first download the 4 sets of xBD images
then put them in the directory as shown below
then move them to another directory called `xBD_images` 1 by 1
then run the commented code separately
"""

import pandas as pd
import os, shutil, re
df = pd.read_csv("data/OpenEarthMap/OpenEarthMap_wo_xBD/xbd_files.csv", header=None)




xbd_image_names = df[0].tolist()
found_image_names = []

# do for all 4
tier3_image_names = os.listdir("data/OpenEarthMap/tier3/tier3/images")
train_image_names = os.listdir("data/OpenEarthMap/train_images_labels_targets/train/images")
test_image_names = os.listdir("data/OpenEarthMap/test_images_labels_targets/test/images")
hold_image_names = os.listdir("data/OpenEarthMap/hold_images_labels_targets/hold/images")

for xbd_image_name in xbd_image_names:
    if xbd_image_name in tier3_image_names:

        source_path = f"data/OpenEarthMap/tier3/tier3/images/{xbd_image_name}"
        destination_path = f"data/OpenEarthMap/xBD_images/{xbd_image_name}"

        shutil.move(source_path, destination_path)




# for i, row in df.iterrows():
#     old_image_name = row[0]
#     new_image_name = row[1]
#     location_name = str(re.search(r'^(.*)_(?=\d)', new_image_name).group(1))
#
#     source_path = f'data/OpenEarthMap/xBD_images/{old_image_name}'
#     destination_path = f'data/OpenEarthMap/OpenEarthMap_wo_xBD/{location_name}/images/{new_image_name}'
#
#     shutil.copy(source_path, destination_path)