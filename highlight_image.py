from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO("runs/detect/train/weights/best.pt")

# Detectar tornillos en una imagen
results = model.predict("dataset/val/images/tornillo7.jpg")

# Mostrar la imagen con el tornillo resaltado
results.show()
