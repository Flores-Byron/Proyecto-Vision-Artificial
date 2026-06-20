from ultralytics import YOLO

# Cargar modelo base
model = YOLO("yolov8n.pt")

# Entrenar
model.train(
    data="/workspaces/Deteccion_de_Placas_Vehiculares/dataset/data.yaml",
    epochs=50,
    imgsz=640
)
