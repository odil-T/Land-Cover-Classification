import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import earthpy.plot as ep
import rasterio as rio
import tqdm
import glob
import fiona
import geopandas as gpd
import pandas as pd
import rasterio.mask
from utils import post_processing
from pathlib import Path
from rasterio import windows
from rasterio.merge import merge
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import unary_union #cascaded_union,
from itertools import product
from scipy import ndimage

def save_detections1(detection_filename, tiff_file, pred):
    pred = pred.squeeze(0).cpu().detach().numpy()
    #pred = pred[4,:,:]
    opened_img = (pred - pred.min()) * ((255 - 0) / (pred.max() - pred.min())) + 0
    #opened_img = np.expand_dims(opened_img, axis=0)
    opened_img = opened_img.astype('uint8')
    #opened_img[opened_img > 0] = 255

    # with rio.open(tiff_file) as src:
    #     arr = src.read()
    src = rio.open(tiff_file[0])
    
    with rio.open(detection_filename,"w",driver=src.driver, count=1, dtype=opened_img.dtype, 
                width=256, height=256, transform=src.transform, crs=src.crs) as raschip:
                raschip.write(opened_img)


def save_image1(img, ndvi, mask, pred, filepath, flag=True):
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = ((img - np.min(img)) * 255 / (np.max(img) - np.min(img))).astype('uint8')

    ndvi = ndvi.squeeze(0).cpu().numpy()
    ndvi = cv2.cvtColor((ndvi * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

    mask = mask.cpu().numpy()
    mask = cv2.cvtColor((mask * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

    pred = torch.clamp(pred, 0., 1.)
    pred = (pred.squeeze(0).cpu().numpy() * 255).astype('uint8')
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    if flag:
        opened_img, number_of_objects, blob_labels = post_processing(pred)

        f, axs = plt.subplots(1,5, figsize=(23,23))

        ep.plot_rgb(np.uint8(img.transpose([2,0,1])),
                    rgb=[0, 1, 2],
                    title="Satellite Image",
                    stretch=True,
                    ax=axs[0])

        axs[1].imshow(mask)
        axs[1].set_title("Ground Truth")

        axs[2].imshow(pred)
        axs[2].set_title("Model Prediction")

        axs[3].imshow(opened_img)
        axs[3].set_title("Post-processed")

        axs[4].imshow(blob_labels, cmap='gist_ncar')
        axs[4].set_title("Number of Trees: {}".format(number_of_objects[-1]))

        plt.savefig(filepath, bbox_inches='tight')
        plt.close('all')
    else:
        final = cv2.hconcat([img[:,:,:3], ndvi, mask, pred])
        cv2.imwrite(filepath, final)

# def save_image(img, mask, pred, filepath, flag=True):
#     img = img.cpu().detach().numpy().transpose((1, 2, 0)) 
#     img = (img - img.min()) * ((255 - 0) / (img.max() - img.min())) + 0
#     img = img.astype('uint8')

#     mask = mask.cpu().detach().numpy() 
#     mask *= 255.
#     mask = mask.astype('uint8')
#     if len(mask.shape) != 3: 
#         mask = cv2.merge((mask, mask, mask))
#     else:
#         mask = mask.transpose((1, 2, 0))

#     pred = torch.clamp(pred, 0., 1.)
#     pred = pred.squeeze(0).cpu().detach().numpy()
#     output = (pred - pred.min()) * ((255 - 0) / (pred.max() - pred.min())) + 0
#     output = output.astype('uint8')
#     output = cv2.merge((output, output, output))
   
#     if flag: 
#         opened_img, number_of_objects, blob_labels = post_processing(pred)
#         #print("Number of Trees: {}".format(number_of_objects[-1]))

#         f, axs = plt.subplots(1,5, figsize=(23,23))

#         ep.plot_rgb(np.uint8(img.transpose([2,0,1])),
#                     rgb=[0, 1, 2],
#                     title="Satellite Image",
#                     stretch=True,
#                     ax=axs[0])

#         axs[1].imshow(np.uint8(mask))
#         axs[1].set_title("Ground Truth")

#         axs[2].imshow(pred)
#         axs[2].set_title("Model Prediction")

#         axs[3].imshow(opened_img)
#         axs[3].set_title("Post-processed")

#         axs[4].imshow(blob_labels, cmap='gist_ncar')
#         axs[4].set_title("Number of Trees: {}".format(number_of_objects[-1]))

#         plt.savefig(filepath, bbox_inches='tight')
#         plt.close('all')
#     else:
#         final = cv2.hconcat([img[:,:,:3], mask, output])
#         cv2.imwrite(filepath, final)

# def save_detections(detection_filename, tiff_file, pred):

#     src = rio.open(tiff_file)
    
#     with rio.open(detection_filename,"w",driver=src.driver, count=1, dtype=pred.dtype, 
#                 width=256, height=256, transform=src.transform, crs=src.crs) as raschip:
#                 raschip.write(pred)


def save_detections(detection_filename, pred, transform, crs):
    
    rows, cols = pred.shape
    
    # Define metadata
    meta = {
        "driver": "GTiff",
        "dtype": pred.dtype,
        "count": 1,
        "height": rows,
        "width": cols,
        "transform": transform,
        "crs": crs
    }
    
    # Write data in chunks
    chunk_size = 1024
    with rio.open(detection_filename, "w", **meta) as out_file:
        for col in range(0, cols, chunk_size):
            out_file.write(pred[:, col:col+chunk_size], window=((0, rows), (col, col+chunk_size)))
            
    print(f"Saved {detection_filename}")
#------------------------------------------
def img_rgb_from_filepath(img_filepath):
    with rio.open(img_filepath) as src:
        arr = src.read()

    return np.rollaxis(arr[:3], 0, 3)

# def tif2png(raster_folder):
#     tiles_folder = raster_folder + "tiles\\"
#     outpath_folder = raster_folder + "tiles_png\\"

#     img_filenames = os.listdir(tiles_folder)
    
#     for i in tqdm.tqdm(range(len(img_filenames))):
#         img = img_rgb_from_filepath(tiles_folder + img_filenames[i])
#         img = (img - img.min()) * ((255 - 0) / (img.max() - img.min())) + 0
#         img = np.uint8(img)
#         outfile = img_filenames[i].split('.')[0] + '.png'
#         cv2.imwrite(outpath_folder + outfile, img)

def tif2png(raster_folder):
    tiles_folder = raster_folder + "tiles\\"
    outpath_folder = raster_folder + "tiles_png\\"

    # Use os.scandir instead of os.listdir
    img_filenames = [entry.name for entry in os.scandir(tiles_folder)]

    # Precompute the maximum and minimum values for the entire dataset
    max_val = -float('inf')
    min_val = float('inf')
    for filename in img_filenames:
        img = img_rgb_from_filepath(tiles_folder + filename)
        max_val = max(max_val, img.max())
        min_val = min(min_val, img.min())

    for i in tqdm.tqdm(range(len(img_filenames))):
        img = img_rgb_from_filepath(tiles_folder + img_filenames[i])
        img = (img - min_val) * ((255 - 0) / (max_val - min_val)) + 0
        img = np.uint8(img)

        # Use a list comprehension to generate the output filenames
        outfile = f"{img_filenames[i].split('.')[0]}.png"
        cv2.imwrite(f"{outpath_folder}{outfile}", img)


# def create_tiles_mod(dir, imgfile, folder_name):
#     os.chdir(dir)

#     # Save tiles into new directory
#     out_path = dir + folder_name
#     output_filename = 'tile_{}-{}.tif'
#     os.chdir(out_path)

#     def get_tiles(ds, width=256, height=256):
#         nols, nrows = ds.meta['width'], ds.meta['height']
#         offsets = product(range(0, nols, width), range(0, nrows, height))
#         big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#         for col_off, row_off in  offsets:
#             window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
#             transform = windows.transform(window, ds.transform)
#             yield window, transform

#     with rio.open(os.path.join(os.path.join(dir, imgfile))) as inds:
#         tile_width, tile_height = 256, 256

#         meta = inds.meta.copy()
#         array = np.zeros((meta['count'], tile_width, tile_height), dtype=np.uint16)

#         for window, transform in get_tiles(inds):
#             print(window)
#             meta['transform'] = transform
#             meta['width'], meta['height'] = tile_width, tile_height #window.width, window.height 
#             outpath = os.path.join(out_path,output_filename.format(int(window.col_off), int(window.row_off)))
#             with rio.open(outpath, 'w', **meta) as outds:
#                 temp = inds.read(window=window)
                
#                 if temp.shape[1] < tile_width or temp.shape[2] < tile_height:
#                     # Pad the smaller array with zeros so that both arrays have the same shape
#                     if temp.shape[0] < array.shape[0]:
#                         temp = np.pad(temp, ((0, array.shape[0]-temp.shape[0]), (0, 0), (0, 0)))
#                     elif temp.shape[0] > array.shape[0]:
#                         array = np.pad(array, ((0, temp.shape[0]-array.shape[0]), (0, 0), (0, 0)))

#                     if temp.shape[1] < array.shape[1]:
#                         temp = np.pad(temp, ((0, 0), (0, array.shape[1]-temp.shape[1]), (0, 0)))
#                     elif temp.shape[1] > array.shape[1]:
#                         array = np.pad(array, ((0, 0), (0, temp.shape[1]-array.shape[1]), (0, 0)))

#                     if temp.shape[2] < array.shape[2]:
#                         temp = np.pad(temp, ((0, 0), (0, 0), (0, array.shape[2]-temp.shape[2])))
#                     elif temp.shape[2] > array.shape[2]:
#                         array = np.pad(array, ((0, 0), (0, 0), (0, temp.shape[2]-array.shape[2])))
                    
#                     temp = array + temp
#                 outds.write(temp)

def create_tiles_mod(dir, imgfile, folder_name):
    os.chdir(dir)

    # Save tiles into new directory
    out_path = os.path.join(dir, folder_name)
    output_filename = 'tile_{}-{}.tif'
    os.makedirs(out_path, exist_ok=True)

    def get_tiles(ds, width=256, height=256):
        nols, nrows = ds.meta['width'], ds.meta['height']
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform

    with rio.open(os.path.join(dir, imgfile)) as inds:
        tile_width, tile_height = 256, 256

        meta = inds.meta.copy()
        meta['width'], meta['height'] = tile_width, tile_height
        meta.pop('transform', None)

        for window, transform in get_tiles(inds):
            outpath = os.path.join(out_path, output_filename.format(int(window.col_off), int(window.row_off)))
            meta['transform'] = transform
            with rio.open(outpath, 'w', **meta) as outds:
                temp = inds.read(window=window)

                if temp.shape[1] < tile_width or temp.shape[2] < tile_height:
                    # Pad the smaller array with zeros so that both arrays have the same shape
                    pad_width = ((0, 0), (0, tile_height - temp.shape[1]), (0, tile_width - temp.shape[2]))
                    temp = np.pad(temp, pad_width)

                outds.write(temp)


def create_tiles(dir, imgfile, folder_name, stride=256, winsize=256, remove_empty=False):
    ## Read raster
    os.chdir(dir)
    raster = rio.open(dir+imgfile)
    imgarr=raster.read()
    print("Raster o'lchami: ", imgarr.shape) 

    index=0
    ## Save tiles into new directory
    outpath = dir + folder_name
    os.chdir(outpath)
    
    # whites = [255,255,255]
    # blacks = [0, 0, 0]

    ## Loop trough raster
    for i in np.arange(0,imgarr.shape[1],stride):
        for j in np.arange(0,imgarr.shape[2],stride):
            out_img = imgarr[:,i:i+winsize,j:j+winsize]
            # Check if frame is empty
            if remove_empty:
                ## Prepocess
                # temp = out_img[:,:,:3]
                # temp = cv2.convertScaleAbs(temp)
                # temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)

                # # Count white pixels
                # whitepx = np.count_nonzero(np.all(temp==255))
                # # Count black pixels
                # blackpx = np.count_nonzero(np.all(temp==0))

                # totalpx = temp.shape[0] * temp.shape[1]
                # whitepercent=(whitepx/totalpx)*100
                # blackpercent=(blackpx/totalpx)*100
                # if whitepercent > 40.0 or  blackpercent > 40.0:
                #     print("Raster bo'sh (ma'nosiz) bo'lgan hududga ega va tashlab ketilmoqda!")
                #     continue
                
                ## Unused code
                # temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
                # temp = cv2.fastNlMeansDenoising(temp, None, 20, 7, 21) 
                # # count = np.count_nonzero(temp==0)
                
                # Working code
                uni, counts = np.unique(out_img, return_counts=True)
                if counts[0] > 50000 or uni.shape == (2,) or np.mean(out_img) == 0.0 or np.mean(out_img) == 65535.0:
                    print("Raster bo'sh (ma'nosiz) bo'lgan hududga ega va tashlab ketilmoqda!")
                    continue

            x,y = (j*raster.transform[0]+raster.transform[2]),(raster.transform[5]+i*raster.transform[4])
            out_img_transform = [raster.transform[0],0,x,0,raster.transform[4],y]
            index+=1
            with rio.open(str(index)+"_img.tif","w",driver=raster.driver, count=imgarr.shape[0], dtype=imgarr.dtype, #'GTiff'
                         width=winsize, height=winsize, transform=out_img_transform, crs=raster.crs) as raschip: #, crio=raster.crio
                         raschip.write(out_img)
                
    print("Tayllarning umumiy soni: ", index)


# def fill_holes(raster_folder, new_mask_folder):
#     mask_folder = raster_folder + 'masks\\'
#     mask_link_list = sorted(glob.glob(mask_folder+"*"))

#     for link in mask_link_list:
#         raster = rio.open(link)
#         imgarr = raster.read()

#         gray = imgarr[0, :, :]
#         gray = (gray - gray.min()) * ((255 - 0) / (gray.max() - gray.min())) + 0
#         gray = gray.astype('uint8')

#         gray = ndimage.binary_fill_holes(gray).astype(int)
#         gray = (gray - gray.min()) * ((255 - 0) / (gray.max() - gray.min())) + 0
#         gray = gray.astype('uint8')
#         gray = np.expand_dims(gray, axis=0)

#         _, tail = os.path.split(link)
#         out_filename = new_mask_folder + tail

#         with rio.open(out_filename,"w",driver=raster.driver, count=1, dtype=gray.dtype, 
#                     width=raster.width, height=raster.height, transform=raster.transform, crs=raster.crs) as raschip:
#                     raschip.write(gray)

def fill_holes(raster_folder, new_mask_folder):
    mask_folder = os.path.join(raster_folder, 'masks')
    mask_link_list = sorted(glob.glob(os.path.join(mask_folder, '*')))

    for link in mask_link_list:
        with rio.open(link) as src:
            imgarr = src.read(1, masked=True)

            # filled = ndimage.binary_fill_holes(imgarr).astype(imgarr.dtype, copy=False)
            filled = ndimage.binary_fill_holes(imgarr).astype(np.uint8, copy=False) * 255

            out_filename = os.path.join(new_mask_folder, os.path.split(link)[1])
            with rio.open(out_filename, 'w', driver=src.driver,
                          count=1, dtype=filled.dtype, width=src.width,
                          height=src.height, transform=src.transform,
                          crs=src.crs) as dst:
                dst.write(filled, 1)

# def create_mosaic(raster_folder):
#     dets = raster_folder + 'detections\\'
#     path = Path(dets)
#     output_folder = raster_folder + "mosaic\\mosaic_output.tif"

#     raster_files = list(path.iterdir())
#     raster_to_mosiac = []

#     for p in raster_files:
#         raster = rio.open(p)
#         raster_to_mosiac.append(raster)

#     mosaic, output = merge(raster_to_mosiac)

#     output_meta = raster.meta.copy()
#     output_meta.update(
#         {"driver": "GTiff",
#             "height": mosaic.shape[1],
#             "width": mosaic.shape[2],
#             "transform": output,
#         }
#     )

#     with rio.open(output_folder, "w", **output_meta) as m:
#         m.write(mosaic)

def create_mosaic(raster_folder):
    dets = os.path.join(raster_folder, 'detections')
    output_folder = os.path.join(raster_folder, 'mosaic', 'mosaic_output.tif')

    raster_files = glob.glob(os.path.join(dets, '*'))
    raster_to_mosaic = [rio.open(p) for p in raster_files]

    mosaic, output = merge(raster_to_mosaic)

    output_meta = raster_to_mosaic[0].meta.copy()
    output_meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': output,
    })

    with rio.open(output_folder, 'w', **output_meta) as m:
        m.write(mosaic)


# def create_geometry(raster_input):
#     """
#     Creates a shapely polygon from a raster image
#     :param raster_input: raster path
#     :return: shapely geometry
#     """
#     mask = None
#     results = None
#     #pixels = []

#     # with rasterio.driverio():
#     with rio.open(raster_input) as src:
#         image = src.read(1)  # firiot band
#         out_crs=src.crs
#         mask = image != 0
#         results = (
#             {'properties': {'raster_val': v}, 'geometry': s}
#             for i, (s, v)
#             in enumerate(
#             shapes(image, mask=mask, transform=src.transform)))

#     return results, out_crs

    # import geopandas as gpd

def create_geometry(raster_input):
    """
    Creates a shapely polygon from a raster image
    :param raster_input: raster path
    :return: shapely geometry
    """
    with rio.open(raster_input) as src:
        image = src.read(1, masked=True)
        out_crs = src.crs
        polygons = shapes(image, transform=src.transform)
        records = [{'geometry': poly, 'raster_val': val} for poly, val in polygons]
        gdf = gpd.GeoDataFrame.from_records(records, crs=out_crs)
    return gdf.geometry, out_crs



# def merge_polygons(geometries):
#     """
#     Merge polygons that are neighbors
#     :param geometries: shapely geometries (polygons)
#     :return: merged polygons
#     """

#     results = []

#     for polygon in geometries:
#         polygons = shape(polygon['geometry'])
#         results.append(polygons)

#     print("Creating union from polygons...")

#     ## Try using GeoPandas
#     # geometry = GeoSeries(polygons)
#     # geometry = geometry.unary_union

#     return unary_union(results) #cascaded_union(results) # ## cascaded_union is deprecated in newer version. MUST USE shapely 1.7!

def merge_polygons(geometries):
    """
    Merge polygons that are neighbors
    :param geometries: shapely geometries (polygons)
    :return: merged polygons
    """
    polygons = [shape(polygon['geometry']) for polygon in geometries]
    return unary_union(polygons)


# def polygonalize_geometry(geometry, out_crs, output_path):
#     """
#     Creates a shapely polygon from the geometry
#     :param geometry: Geometry from create_geometry()
#     :return: a shapely polygon
#     """

#     output_file = os.path.join(os.path.dirname(output_path), "image_features.shp")
#     print("Writing file {}".format(output_file)) #output_path
#     out_schema = {
#         'geometry': 'Polygon',
#         'properties': {'id': 'int'}
#     }

#     # write the shapefile
#     with fiona.open(output_file, 'w', crs=out_crs.data, driver='ESRI Shapefile', schema=out_schema) as c:
#         for polygon in geometry.geoms:
#             c.write({
#                 'geometry': mapping(shape(polygon)), #polygon['geom_type']
#                 'properties': {'id': 123}
#             })

out_schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'}
}

def polygonalize_geometry(geometry, out_crs, output_path):
    """
    Creates a shapely polygon from the geometry
    :param geometry: Geometry from create_geometry()
    :return: a shapely polygon
    """

    print("Writing file {}".format(output_path))
    with fiona.open(output_path, 'w', crs=out_crs, driver='ESRI Shapefile', schema=out_schema) as c:
        for polygon in geometry.geoms:
            c.write({
                'geometry': mapping(shape(polygon)),
                'properties': {'id': 123}
            })



def merge_polygons_whole(raster_folder):
    vector_folder = raster_folder + "vector\\"
    file = os.listdir(vector_folder)
    path = [os.path.join(vector_folder, i) for i in file if ".shp" in i]
    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in path], 
                            ignore_index=True), crs=gpd.read_file(path[0]).crs)
    gdf.to_file(vector_folder+'compiled.shp')

def shape_builder(raster_input, output_path):
    "Builds a polygon out of the pixel polygons"

    # polygons = []
    polygons, out_crs = list(create_geometry(raster_input))
    polygons = merge_polygons(polygons)
    polygonalize_geometry(polygons, out_crs, output_path)


def vectorize(raster_folder):
    print("Geometriya hisoblanmoqda...")

    mosaic_file = raster_folder + "mosaic\\mosaic_output.tif"
    vector_folder = raster_folder + "vector\\"
    polygon = shape_builder(mosaic_file, vector_folder)
    # print(polygon)


def detections2polygons(raster_input, output_filename):
    # polygons = list(create_geometry(raster_input))
    # polygonalize_geometry(polygons, output_filename)

    polygon = create_geometry(raster_input)
    polygonalize_geometry(polygon, output_filename)

# def shape2mask(raster_folder, raster_file, shape_file, mask_file):
#     with fiona.open(raster_folder+shape_file, "r") as shapefile:
#         shapes = [feature["geometry"] for feature in shapefile]

#     with rasterio.open(raster_folder+raster_file) as src:
#         out_image, out_transform = rasterio.mask.mask(src, shapes, filled=True) #crop=True,
#         out_meta = src.meta

#     out_meta.update({"driver": "GTiff",
#                     "height": out_image.shape[1],
#                     "width": out_image.shape[2],
#                     "transform": out_transform})

#     with rasterio.open(raster_folder+mask_file, "w", **out_meta) as dest:
#         dest.write(out_image)

# def shape2mask(raster_folder, raster_file, shape_file, mask_file):
#     with fiona.open(os.path.join(raster_folder, shape_file), "r") as shapefile:
#         shapes = [feature["geometry"] for feature in shapefile]

#     with rasterio.open(os.path.join(raster_folder, raster_file)) as src:
#         out_image, out_transform = rasterio.mask.mask(src, shapes, filled=True) #crop=True,
#         out_meta = src.meta.copy()

#     out_meta.update({"driver": "GTiff",
#                     "height": out_image.shape[1],
#                     "width": out_image.shape[2],
#                     "transform": out_transform})

#     with rasterio.open(os.path.join(raster_folder, mask_file), "w", **out_meta) as dest:
#         dest.write(out_image)

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


