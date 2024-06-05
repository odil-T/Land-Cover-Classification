from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt


path = r"datasets/SN2_buildings_train_AOI_5_Khartoum\AOI_5_Khartoum_Train\RGB-PanSharpen\RGB-PanSharpen_AOI_5_Khartoum_img155.tif"

dataset = gdal.Open(path, gdal.GA_ReadOnly)

# width = dataset.RasterXSize
# height = dataset.RasterYSize

scaled_bands = []
for band_index in range(1, dataset.RasterCount + 1):
    band = dataset.GetRasterBand(band_index).ReadAsArray()
    scaled_band = (band - band.min()) / (band.max() - band.min())
    scaled_bands.append(scaled_band)
image = np.stack(scaled_bands, axis=-1)

dataset = None

# image is ready to be used