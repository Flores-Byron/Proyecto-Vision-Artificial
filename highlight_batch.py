from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# Procesar todas las imágenes de validación y guardar resultados
results = model.predict("dataset/val/images", save=True)

# Las imágenes con rectángulos se guardan en runs/detect/predict/
