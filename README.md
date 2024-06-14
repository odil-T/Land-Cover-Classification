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

### How to use the pretrained model [INCOMPLETE]

You can either use Docker or use the source directly.

#### With Docker

#### From Source

1. Ensure you have [Python 3.11](https://www.python.org/downloads/release/python-3119/), [Git](https://git-scm.com/), and [Anaconda](https://www.anaconda.com/products/individual) installed on your system before running the setup.
2. Open a terminal and clone the repository with `git clone https://github.com/odil-T/Land-Cover-Classification`.

Please provide 650 x 650 images with 0.3m GSD resolution.


### Additional Information

The `semantic-segmentation` directory contains the files to train and infer the semantic segmentation model.

The `instance-segmentation` directory contains the files to train and infer the instance segmentation model. I used some code from [rcland12's detectron2-spacenet repository](https://github.com/rcland12/detectron2-spacenet).

The `object-detection` directory contains the files that were used to train and infer an object detection model that detected cars and buildings from aerial images. The object detection was needed in order to input the predicted bounding boxes to SAM to perform instance segmentation of cars and buildings. However this model is not used because it did not perform well on real satellite images and, therefore, has been replaced by the instance segmentation model.

### How to access the pretrained models

The models are stored in [my HuggingFace repository](https://huggingface.co/odil111).

The final models used are the following:
- [SegFormer model for Semantic Segmentation](https://huggingface.co/odil111/segformer-fine-tuned-on-openearthmap/tree/main/segformer_sem_seg_2024-06-05--16-54-31)
- [YOLOv8 model for Instance Segmentation](https://huggingface.co/odil111/yolov8m-seg-fine-tuned-on-spacenetv2/tree/main/yolov8m_inst_seg_2024-06-11--15-57-15/weights)
