from ultralytics import YOLO

# Cargar modelo preentrenado
model = YOLO("yolov8n.pt")

# Entrenar con tu dataset
model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640
)

# Probar el modelo
results = model.predict("dataset/val/images")
results.show()
