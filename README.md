# Land-Cover-Classification

This repository contains the files necessary to run inference and training of models to perform panoptic segmentation of satellite images for land cover classification.

The panoptic segmentation is performed by using semantic and instance segmentation models. The semantic segmentation model first performs segmentation on the image to predict the masks of different classes on the image. Thereafter, the instance segmentation model predicts instance masks of the `Building` class and overlays them on the semantic mask.

The semantic model was trained on the [OpenEarthMap dataset](https://open-earth-map.org/). The instance model was trained on the [SpaceNet-v2 dataset](https://spacenet.ai/spacenet-buildings-dataset-v2/).

The semantic model can detect the following classes from the image:
- `Bareland`
- `Rangeland`
- `Developed space`
- `Road`
- `Tree`
- `Water`
- `Agriculture land`
- `Building`

The instance model only detects instances of the `Building` class.

Here is a sample screenshot from the Streamlit app:

![image](https://github.com/odil-T/Land-Cover-Classification/assets/142138394/f6eb1410-b4a0-4748-ac16-5c45dffdf6e2)


### How to use the app [INCOMPLETE]

You can either use Docker or install from source directly.

#### With Docker
1. Ensure you have Docker installed.
2. Open a terminal and enter `docker pull odil713/land-cover-classification:latest`.
3. Run the Docker Image with `docker run -p 8501:8501 lulc`.
4. Enter `localhost:8501` in your browser to open the app.

#### From Source

1. Ensure you have [Git](https://git-scm.com/) and [Anaconda](https://www.anaconda.com/download/success) installed on your system before running the setup.
2. Open a terminal and clone the repository with `git clone https://github.com/odil-T/Land-Cover-Classification`.
3. Switch to the local repository's directory by entering `cd Land-Cover-Classification`.
4. Setup the environment by running `./setup.bat` for Windows or `sh setup.bat` for Linux. This will take some time.
5. While the setup is running, you can download the pretrained [SegFormer](https://huggingface.co/odil111/segformer-fine-tuned-on-openearthmap/blob/main/segformer_sem_seg_2024-06-05--16-54-31/segformer_sem_seg_checkpoint_epoch35.pt) and [YOLO](https://huggingface.co/odil111/yolov8m-seg-fine-tuned-on-spacenetv2/blob/main/yolov8m_inst_seg_2024-06-11--15-57-15/weights/best.pt) models and place them in `best_models` directory.
6. Launch the app by running `conda activate lulc` followed by `streamlit run app.py`. A new window should appear in your browser.
7. Just close the terminal to close the app.
8. If you wish to reopen the app, then open the terminal, navigate to the local repository's directory with `cd PATH/TO/YOUR/Land-Cover-Classification` and repeat step 5.

Provide 650 x 650 images with ~0.3m GSD resolution. The image dimensions do not strictly have to be 650 x 650. If the provided image dimensions are larger, the image center will be cropped and used.


### Additional Information

The `semantic-segmentation` directory contains the files to train and infer the semantic segmentation model.

The `instance-segmentation` directory contains the files to train and infer the instance segmentation model. I used some code from [rcland12's detectron2-spacenet repository](https://github.com/rcland12/detectron2-spacenet).

The `object-detection` directory contains the files that were used to train and infer an object detection model that detected cars and buildings from aerial images. The object detection was needed in order to input the predicted bounding boxes to SAM to perform instance segmentation of cars and buildings. However this model is not used because it did not perform well on real satellite images and, therefore, has been replaced by the instance segmentation model.

### How to access the pretrained models

The models are stored in [my HuggingFace repository](https://huggingface.co/odil111).

The final models used are the following:
- [SegFormer model for Semantic Segmentation](https://huggingface.co/odil111/segformer-fine-tuned-on-openearthmap/blob/main/segformer_sem_seg_2024-06-05--16-54-31/segformer_sem_seg_checkpoint_epoch35.pt)
- [YOLOv8 model for Instance Segmentation](https://huggingface.co/odil111/yolov8m-seg-fine-tuned-on-spacenetv2/blob/main/yolov8m_inst_seg_2024-06-11--15-57-15/weights/best.pt)
