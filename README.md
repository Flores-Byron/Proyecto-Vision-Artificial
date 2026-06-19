# Detección de Placas de Coches con YOLOv8

## Integrante
- Byron Zadquiel Flores Gonzalez-23310338

## Instrucciones para ejecutar
```bash
git clone https://github.com/tu_usuario/Deteccion-Placas-YOLO.git
cd Deteccion-Placas-YOLO
pip install -r requirements.txt
python train.py

Caso de Estudio: Detección de Placas Vehiculares con YOLOv8
1. Planteamiento del problema
En sistemas de control de acceso vehicular y vigilancia urbana, la identificación automática de placas de coches es esencial para mejorar la seguridad, agilizar procesos y reducir errores humanos. Actualmente, muchos estacionamientos y carreteras dependen de operadores o sistemas poco precisos.

Problema: ¿Cómo implementar un sistema de visión artificial que detecte placas de coches en tiempo real, con alta precisión y bajo costo?

2. Objetivo del proyecto
Desarrollar un modelo de detección de placas vehiculares utilizando YOLOv8, capaz de identificar placas en imágenes y videos en tiempo real, integrable en sistemas de control de acceso o monitoreo.

3. Metodología:
Recolección de datos
*Dataset CCPD (Chinese City Parking Dataset) o imágenes propias de coches en México.
*Organización en carpetas: train/, val/ con imágenes y etiquetas.

Preprocesamiento
*Anotación de placas con bounding boxes (herramientas como LabelImg o Roboflow).
*Conversión a formato YOLO (.txt con coordenadas normalizadas).

Entrenamiento del modelo
*Uso de YOLOv8n (modelo ligero).
*Configuración en data.yaml:

train: dataset/train/images
val: dataset/val/images
nc: 1
names: ["placa"]

*Script train.py:

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="dataset/data.yaml", epochs=50, imgsz=640)

Validación:
*Evaluación en imágenes de prueba.
*Métricas: precisión, recall, mAP.

Implementación en tiempo real:
*Integración con OpenCV para capturar video de cámara:

python
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model.predict(frame)
    annotated = results[0].plot()
    cv2.imshow("Detección de Placas", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

4. Resultados esperados
*Detección precisa de placas en diferentes condiciones de luz y ángulos.
*Reducción de errores en sistemas de control vehicular.
*Posibilidad de integrar OCR (Reconocimiento Óptico de Caracteres) para leer el texto de la placa.

5. Aplicaciones
*Control de acceso en estacionamientos.
*Registro automático de vehículos en peajes.
*Vigilancia urbana para seguridad pública.
*Sistemas de rastreo y logística.