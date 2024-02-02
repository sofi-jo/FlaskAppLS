# Importamos librerias
import cv2
import os
from ultralytics import YOLO

# Importamos clase de detección de manos
from utils import seguimientoManos as sm
# Lectura del modelo
model = YOLO("models/lenguasen.pt")

# Lectura de camara
cap = cv2.VideoCapture(1)

# Cambio resolucion
cap.set(3, 1280)
cap.set(4, 720)

# Declarar detector manos
detector = sm.detectormanos(Confdeteccion=0.8)

# Empieza programa
while True:
    # Lectura videocaptura
    ret, frame = cap.read()

    # Encontrar manos
    frame = detector.encontrarmanos(frame, dibujar=False)

    # Posiciones mano
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False, color=[0, 255, 0])

    if mano == 1:
        # Extrae informacion del cuadro
        xmin, ymin, xmax, ymax = bbox
        # Asigna  margen
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        recorte = frame[ymin:ymax, xmin:xmax]

        # Estandarizar redimensionamiento
        recorte = cv2.resize(recorte, (720, 720), interpolation=cv2.INTER_CUBIC)

        # Extraer resultados
        resultados = model.predict(recorte, conf=0.55)

        # Si existe resultado
        if len(resultados) != 0:
            # Iteramos
            for result in resultados:
                masks = result.masks
                coordenadas = masks

                anotaciones = resultados[0].plot()

        cv2.imshow("RECORTE", anotaciones)

    # Mostrar FSP
    cv2.imshow("LENGUAJE SENAS", frame)

    #LEER TECLADO
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()



