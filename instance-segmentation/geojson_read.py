import geojson
from osgeo import gdal


image_height = 650
image_width = 650

geojson_path = 'datasets/SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train/geojson/buildings/buildings_AOI_5_Khartoum_img1.geojson'
image_path = 'datasets/SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train/RGB-PanSharpen/RGB-PanSharpen_AOI_5_Khartoum_img1.tif'

with open(geojson_path) as f:
    gj = geojson.load(f)

num_buildings = len(gj["features"])

with open("AOI_5_Khartoum_img1.txt", "w") as f:

    if num_buildings > 0:
        gdal_image = gdal.Open(image_path)  # must be image file path

        pixel_width, pixel_height = gdal_image.GetGeoTransform()[1], gdal_image.GetGeoTransform()[5]
        originX, originY = gdal_image.GetGeoTransform()[0], gdal_image.GetGeoTransform()[3]

        for i in range(num_buildings):
            f.write("0 ")

            points = gj["features"][i]["geometry"]["coordinates"][0]
            if len(points) == 1:
                points = points[0]

            for j in range(len(points)):
                point_x = int(round((points[j][0] - originX) / pixel_width)) / image_width  # normalizing
                point_y = int(round((points[j][1] - originY) / pixel_height)) / image_height

                f.write(f"{point_x:.3f} {point_y:.3f} ")

            f.write("\n")

