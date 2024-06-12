# Land-Cover-Classification

This repository contains the files necessary to run inference and training of models to perform panoptic segmentation of satellite images for land cover classification.

The panoptic segmentation is performed by using semantic and instance segmentation models. The semantic segmentation model first performs segmentation on the image to predict the masks of different classes on the image. Thereafter, the instance segmentation model predicts instance masks of the `building` class and overlays them on the semantic mask.










The `semantic-segmentation` directory contains the files to train and infer the semantic segmentation model.

The `instance-segmentation` directory contains the files to train and infer the instance segmentation model. I used some code from [rcland12's detectron2-spacenet repository](https://github.com/rcland12/detectron2-spacenet).

The `object-detection` directory contains the files that were used to train and infer an object detection model that detected cars and buildings from aerial images. The object detection was needed in order to input the predicted bounding boxes to SAM to perform instance segmentation of cars and buildings. However this model model is not used anymore because it has been replaced by the instance segmentation model.
