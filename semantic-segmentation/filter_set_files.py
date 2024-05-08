"""
Makes new txt files for train, validation, and test sets where xBD image and mask names are excluded.
The xBD image and mask names are given in the `xbd_files.csv` file.
"""

import os
import pandas as pd


root_data_dir = "data/OpenEarthMap/OpenEarthMap_wo_xBD"

df = pd.read_csv(os.path.join(root_data_dir, "xbd_files.csv"), header=None).iloc[:, 1]

exclude_locations = df.str.extract(r'^(.*)_(?=\d)', expand=False)
exclude_locations = exclude_locations.unique()

for set_ in ("train", "val", "test"):

    with open(f"{root_data_dir}/{set_}.txt", "r") as f:
        filtered_list = [location for location in f.readlines() if not any(pattern in location for pattern in exclude_locations)]

    with open(f"{root_data_dir}/{set_}_wo_xBD.txt", "w") as f:
        for location in filtered_list:
            f.write(location)
