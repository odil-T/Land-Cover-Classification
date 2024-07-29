import os
import fiona
import rasterio


def shape2mask(raster_folder, raster_file, shape_file, mask_file):
    shape_path = os.path.join(raster_folder, shape_file)
    raster_path = os.path.join(raster_folder, raster_file)
    mask_path = os.path.join(raster_folder, mask_file)

    with fiona.open(shape_path, "r") as shapefile, \
         rasterio.open(raster_path) as src, \
         rasterio.open(mask_path, "w", **src.meta) as dest:

        shapes = [feature["geometry"] for feature in shapefile]
        out_image, out_transform = rasterio.mask.mask(src, shapes, filled=True)

        dest.write(out_image)

        dest_meta = dest.meta.copy()
        dest_meta.update(height=out_image.shape[1], width=out_image.shape[2], transform=out_transform, driver='GTiff')
        dest.write(out_image)



# ChatGPT suggestion -----------------------------------------------------------------------
import os
import rasterio
import rasterio.features
import rasterio.mask
from shapely.geometry import shape, mapping
import fiona
from fiona.crs import from_epsg

def mask2shape(raster_folder, mask_file, shape_file, epsg_code):
    mask_path = os.path.join(raster_folder, mask_file)
    shape_path = os.path.join(raster_folder, shape_file)

    # Read the mask file
    with rasterio.open(mask_path) as src:
        image = src.read(1)  # Read the first band, adjust if multiple bands
        mask = image != 0  # Assuming non-zero values are the mask

        # Extract shapes (polygons) from the raster
        shapes = rasterio.features.shapes(image, mask=mask, transform=src.transform)

        # Define schema for shapefile
        schema = {
            'geometry': 'Polygon',
            'properties': {'value': 'int'}
        }

        # Write shapes to shapefile
        with fiona.open(shape_path, 'w', driver='ESRI Shapefile', crs=from_epsg(epsg_code), schema=schema) as shp:
            for geom, value in shapes:
                if value != 0:  # Write only non-zero shapes
                    shp.write({
                        'geometry': mapping(shape(geom)),
                        'properties': {'value': int(value)}
                    })

# Example usage:
# mask2shape('path/to/raster_folder', 'mask.tif', 'output_shape.shp', 4326)
