from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict("dataset/val/images")
results.show()
