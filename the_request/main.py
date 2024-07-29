import torch
import os
import tqdm
import glob
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from nets.tree_network import*
from loader import TifDataset
from functionals import*
#from opt_functionals import*
from utils import*


## workaround to ensure matplotlib works w/o problems
## if plt is not used, comment this line
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

## Initialize CUDA device if available
## CUDA allows faster processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    ########################## Globals ##########################
    # ## Number of classes for prediction (only tree = 1)
    # NUM_CLASSES = 1
    ## Specify paths to your files (glob glob stores paths to all files in a folder in a list)
    #root = os.getcwd()

    ########################## Folder list ##########################
    print("Ma'lumotlarni saqlash joylari shakllandi.")
    tiles_folder = args.raster_folder + "tiles\\"
    if not os.path.exists(tiles_folder):
        os.mkdir(tiles_folder)

    if args.shape_file != None:
        mask_folder = args.raster_folder + "masks\\"
        if not os.path.exists(mask_folder):
            os.mkdir(mask_folder)

    detections_folder = args.raster_folder + "detections\\"
    if not os.path.exists(detections_folder):
        os.mkdir(detections_folder)

    mosaic_folder = args.raster_folder + "mosaic\\"
    if not os.path.exists(mosaic_folder):
        os.mkdir(mosaic_folder)

    vector_folder = args.raster_folder + "vector\\"
    if not os.path.exists(vector_folder):
        os.mkdir(vector_folder)

    ########################## Shape to Mask processing ##########################
    if args.shape_file != None:
        print("1. Shape faylni maska ko'rinishiga o'tkazish boshlandi!")
        shape2mask(args.raster_folder, args.raster_file, args.shape_file, args.mask_file)
        print("--- Shape faylni maska ko'rinishiga o'tkazish yakunlandi!")

    ########################## Raster and Masks to Tile processing ##########################
    ## function optimized to run on GPU 
    if args.raster2tiles:
        print("2. Raster tayllarga bo'linishi boshlandi!")
        #create_tiles(args.raster_folder, args.raster_file, "tiles", stride=256, winsize=256, remove_empty=False)
        create_tiles_mod(args.raster_folder, args.raster_file, "tiles")
        print("--- Raster tayllarga bo'linishi yakunladi!")

        # print("2. Maska tayllarga bo'linishi boshlandi!")
        # create_tiles(raster_folder, mask_file, "masks", stride=256, winsize=256, pass_empty=False)
        # print("--- Maska tayllarga bo'linishi yakunladi!")
        #create_tiles_mod(args.raster_folder, args.mask_file, "masks")

    ########################## OPTIONAL ##########################
    ########################## Convert tiles into png format ##########################
    if args.tif2png:
        print("--- Tayllarni PNG formatga o'tkazish boshlandi!")
        tif2png(args.raster_folder)
        print("--- Tayllarni PNG formatga o'tkazish yakunlandi!")

    ########################## OPTIONAL ##########################
    ########################## Fill holes in masks ##########################
    if args.fill_holes:
        print("--- Maskalarni to`ldirish boshlandi!")
        new_mask_folder = args.raster_folder + "filled_masks\\"
        if not os.path.exists(new_mask_folder):
            os.mkdir(new_mask_folder)
        fill_holes(args.raster_folder, new_mask_folder)
        print("--- Maskalarni to`ldirish yakunlandi!")


    ########################## Load Models ##########################
    if args.model_path != None:
        print("3. Sun`iy intellekt model yuklanmoqda...")
        model = torch.load(args.model_path).to(DEVICE)
        model.eval()

        # Train_Model = {'Comp_Atten_Unet': Comprehensive_Atten_Unet}
        # model = Train_Model['Comp_Atten_Unet'](args, args.num_input, args.num_classes)
        # model.load_state_dict(torch.load(args.model_path))
        # model.to(DEVICE)
        # model.eval()


        # dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
        # input_names = [ "actual_input" ]
        # output_names = [ "output" ]

        # torch.onnx.export(model,
        #                 dummy_input,
        #                 "D:\\Framework\\onnx\\model_186_cuda_onnx.onnx",
        #                 verbose=False,
        #                 input_names=input_names,
        #                 output_names=output_names,
        #                 export_params=True,
        #                 )

        print("--- Sun`iy intellekt model yuklandi!")
    else:
        print("Sun`iy intellekt modeli topilmadi!")

    ########################## Sun`iy intellekt ##########################
    if args.detecting:
        print("4. Sun`iy intellekt orqali daraxt topish boshlandi!")
        ## Read tiles in tif for ML
        tile_img_link_list = sorted(glob.glob(tiles_folder+"*"))
        tile_data = TifDataset(tile_img_link_list, None, transformation=False, train=False)
        tile_loader = DataLoader(tile_data, batch_size=1, shuffle=False)

        ## Make prediction for each tile
        for _, batch in enumerate(tqdm.tqdm(tile_loader)):
            img_batch, tiff_file = batch
            img_batch = img_batch.to(DEVICE)

            ## predict and save all in png
            pred = model(img_batch)
            pred = pred.squeeze(0).cpu().detach().numpy()
            pred = (pred - pred.min()) * ((255 - 0) / (pred.max() - pred.min())) + 0
            pred = pred.astype('uint8')

            ## First thresholding
            pred[pred<100] = 0

            # ## Second thresholding
            Zlabeled,Nlabels = ndimage.measurements.label(pred)
            label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
            for label, size in enumerate(label_size):
                if size < 40:
                    pred[Zlabeled == label] = 0

            pred[pred>0] = 255

            ## save in tif with georeferencing for ArcGIS Pro
            _, tail = os.path.split(tiff_file[0])
            detection_filename = detections_folder + tail
            save_detections(detection_filename, tiff_file[0], pred)  

        print("--- Sun`iy intellekt orqali daraxt topish yakunlandi!")

    ########################## Detections to tif mosaicing ##########################
    if args.mosaicking:
        print("5. Topilgan daraxtlar mozaikasini qurish boshlandi!")
        create_mosaic(args.raster_folder)
        print("--- Topilgan daraxtlar mozaikasini qurish yakunlandi!")

    ########################## tif mosaicing to vector ##########################
    if args.vectorization:
        print("6. Shape faylga o'tkazish boshlandi!")
        vectorize(args.raster_folder)
        #merge_polygons_whole(args.raster_folder)
        print("--- Shape faylga o'tkazish yakunlandi!")

    ########################## TUGADI! HALAS! TAMAM! KONEC! END! ##########################
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Daraxtlarni topish yakunlandi =", current_time)


if __name__ == "__main__":
        ########################## START ##########################
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Daraxtlarni topish boshlandi =", current_time)
    ########################## Please set the path! ##########################
    parser = argparse.ArgumentParser(description='------------Tree Detection------------')
    # Input related arguments
    parser.add_argument('--raster_folder', default="E:\\Rasters\\Forest\\Samarkand\\4010_002\\",
                        help='Raster fayl joylashgan papka')
    parser.add_argument('--raster_file', default="mos_sam_4010_002.tif",
                        help='Raster fayl nomi')  
    parser.add_argument('--shape_file', default=None,
                        help='Rasterning shape fayl nomi')     
    parser.add_argument('--mask_file', default="whole_mask.tif",
                        help='Rasterning shape fayl nomi')   
    # parser.add_argument('--model_path', default="D:\\Framework\\Non_ndvi_working\\models\\model_186.pt", #"D:\\Tree\\DeepLabV3\\non_ndvi_models\\model_33.pt"
    #                     help='Sun`iy intellekt model joylashgan joyi va nomi')  
    parser.add_argument('--model_path', default="D:\\Framework\\models\\model_250.pt", #"D:\\Tree\\DeepLabV3\\non_ndvi_models\\model_33.pt"
                        help='Sun`iy intellekt model joylashgan joyi va nomi')  
    # parser.add_argument('--model_path', default="D:\\Tree\\DeepLabV3\\non_ndvi_models\\33.pt", #
    #                     help='Sun`iy intellekt model joylashgan joyi va nomi')  

    ## Optional                         
    parser.add_argument('--shape2mask', default=False, type=bool,
                        help='Shape faylni maska ko`rinishiga o`tkazish')
    parser.add_argument('--tif2png', default=False, type=bool,
                        help='Raster tayllarga bo`linishi')  
    parser.add_argument('--fill_holes', default=False, type=bool,
                        help='Maskalarni to`ldirish')       
    
    ## Mandatory - Majburiy
    parser.add_argument('--raster2tiles', default=True, type=bool,
                        help='Raster tayllarga bo`linishi')
    parser.add_argument('--detecting', default=True, type=bool,
                        help='Sun`iy intellekt orqali daraxt topish')
    parser.add_argument('--mosaicking', default=True, type=bool,
                        help='Topilgan daraxtlar mozaikasini qurish')
    parser.add_argument('--vectorization', default=True, type=bool,
                        help='Topilgan daraxtlar mozaikasini qurish')       


    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')  
    parser.add_argument('--out_size', default=(256, 256), help='the output image size')          

    args = parser.parse_args() #args = vars(parser.parse_args())

    main(args)
