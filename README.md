# Land-Cover-Classification

This project is currenty in progress.

The idea of the project is to perform panoptic segmentation on satellite images to perform land cover classification. The land cover can be classified with a semantic segmentation model. The instance segmentation of trees and buildings can be performed by a separate model. The resulting outputs can be overlayed for the final output.

The `semantic-segmentation` directory contains the files to train and infer the semantic segmentation model.

The `instance-segmentation` directory contains the files to train and infer the instance segmentation model. I used some code from [this repository](https://github.com/rcland12/detectron2-spacenet).

The `object-detection` directory contains the files that were used to train and infer an object detection model that detected cars and buildings from aerial images. The object detection was needed in order to input the predicted bounding boxes to SAM to perform instance segmentation of cars and buildings. However this model model is not used anymore because it has been replaced by the instance segmentation model.
