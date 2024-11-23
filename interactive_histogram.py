import cv2
import numpy as np
import matplotlib.pyplot as plt

# Variables globales
drawing = False  # True mientras el usuario está dibujando
ix, iy = -1, -1  # Coordenadas iniciales del rectángulo
rect = None  # Rectángulo final (x, y, w, h)

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect, image_copy

    try:
        # Iniciar el dibujo
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        # Dibujar el rectángulo mientras el botón izquierdo está presionado
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                image_copy = image.copy()
                cv2.rectangle(image_copy, (ix, iy), (x, y), (255, 0, 0), 2)

        # Finalizar el dibujo
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect = (ix, iy, x, y)  # Guardar las coordenadas del rectángulo
            image_copy = image.copy()
            cv2.rectangle(image_copy, (ix, iy), (x, y), (255, 0, 0), 2)
            update_histogram()
    except Exception as e:
        print(f"Error en draw_rectangle: {e}")

def update_histogram():
    global rect
    try:
        if rect is not None:
            x1, y1, x2, y2 = rect
            roi = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]  # ROI en la imagen
            if roi.size > 0:  # Validar que el ROI no esté vacío
                histogram = cv2.calcHist([roi], [0], None, [256], [0, 256])
                average_gray_level = np.mean(roi)
                max_frequency_level = np.argmax(histogram)

                print(f"Promedio del nivel de gris: {average_gray_level:.2f}")
                print(f"Nivel de gris más frecuente: {max_frequency_level} (Frecuencia: {int(histogram[max_frequency_level])})")

                # Graficar el histograma
                plt.clf()
                plt.title("Histograma de la Región de Interés")
                plt.xlabel("Nivel de gris")
                plt.ylabel("Frecuencia")
                plt.plot(histogram)
                plt.xlim([0, 256])
                plt.draw()
                plt.pause(0.001)
    except Exception as e:
        print(f"Error en update_histogram: {e}")

# Cargar la imagen en escala de grises
image = cv2.imread('Estudio_1_png/BOULLOSA-MADRAZO-ANTONIO.transformed160.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta.")

image_copy = image.copy()

# Configurar la ventana interactiva
cv2.namedWindow('Seleccionar ROI')
cv2.setMouseCallback('Seleccionar ROI', draw_rectangle)

plt.ion()  # Activar modo interactivo para Matplotlib
plt.figure()

try:
    while True:
        cv2.imshow('Seleccionar ROI', image_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Presionar 'ESC' para salir
            break
except KeyboardInterrupt:
    print("Salida forzada por el usuario.")
except Exception as e:
    print(f"Error en el bucle principal: {e}")
finally:
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

