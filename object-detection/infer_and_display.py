from ultralytics import YOLO


model = YOLO("models/yolov9c_obj_det_2024-05-09--00-26-34/weights/best.pt")
image_path = "data/test_yolo/images/img_1_0_1552040113157541000.png"

results = model(image_path)

print(results)

for result in results:
    boxes = result.boxes
    result.show()
