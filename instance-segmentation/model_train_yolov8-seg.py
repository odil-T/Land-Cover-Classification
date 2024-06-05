"""
Trains and saves a YOLOv8 model on SpaceNet-v2 dataset for building instance segmentation.
"""

import datetime
from ultralytics import YOLO


current_datetime = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

model = YOLO("yolov8m-seg.pt")

if __name__ == "__main__":
    results = model.train(data="dataset_config.yaml",
                          epochs=100,
                          imgsz=650,
                          device=0,
                          patience=10,
                          batch=-1,
                          save_period=5,
                          project="models",
                          name=f"yolov9c_obj_det_{current_datetime}",
                          seed=42,
                          cos_lr=True,
                          plots=True)
