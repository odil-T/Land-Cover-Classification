from osgeo import ogr
from osgeo import gdal


tif_path = 'Tashkent_images/j_10_030_mask.tif'
shp_path = 'Tashkent_images/j_10_030_mask.shp'


def tif_to_shapefile(tif_path, shp_path):
    src_ds = gdal.Open(tif_path)
    src_band = src_ds.GetRasterBand(1)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = driver.CreateDataSource(shp_path)
    dst_layer = dst_ds.CreateLayer("mask", srs=None)
    new_field = ogr.FieldDefn('ID', ogr.OFTInteger)
    dst_layer.CreateField(new_field)
    gdal.Polygonize(src_band, None, dst_layer, 0, [], callback=None)

    dst_ds = None
    src_ds = None


tif_to_shapefile(tif_path, shp_path)
