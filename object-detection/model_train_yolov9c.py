"""

"""

import os
import dotenv
import datetime
from ultralytics import YOLO


dotenv.load_dotenv()

height = int(os.getenv("TARGET_HEIGHT"))
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

model = YOLO("yolov9c.pt")

results = model.train(data="dataset_config.yaml",
                      epochs=50,
                      imgsz=height,
                      device=0,
                      patience=10,
                      batch=-1,
                      save_period=5,
                      project="models",
                      name=f"yolov9c_obj_det_{current_datetime}",
                      seed=42,
                      cos_lr=True,
                      plots=True)
