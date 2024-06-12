"""
Trains and saves a YOLOv9 model on VALID for object detection.
"""

import datetime
from ultralytics import YOLO


target_size = 650
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

model = YOLO("yolov9c.pt")

if __name__ == "__main__":
    results = model.train(data="dataset_config.yaml",
                          epochs=100,
                          imgsz=target_size,
                          device=0,
                          patience=10,
                          batch=-1,
                          save_period=5,
                          project="models",
                          name=f"yolov9c_obj_det_{current_datetime}",
                          seed=42,
                          cos_lr=True,
                          plots=True)
