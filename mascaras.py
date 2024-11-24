import cv2
import numpy as np
import matplotlib.pyplot as plt

def erotion_dilation(img, kernel, threshold):

    # IMagen blur
    img_filtered = cv2.filter2D(img, -1, kernel)

    # Cobertir a binario
    _, img_binary = cv2.threshold(img_filtered, threshold, 255, cv2.THRESH_BINARY)

    # Aplicar erosion
    erotion = cv2.erode(img_binary, kernel, iterations=1)

    # Aplicar dilatación
    dilation = cv2.dilate(img_binary, kernel, iterations=1)

    return erotion, dilation

def opening_closing(img, thresold, kernel_size):

    # Binarizar la imagen
    _, binary = cv2.threshold(img, thresold, 255, cv2.THRESH_BINARY)

    # Crear un kernel (estructura para las transformaciones morfológicas)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Aplicar operaciones morfológicas
    # 1. Apertura para eliminar ruido
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 2. Cierre para cerrar pequeños agujeros en las regiones segmentadas
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Extraer la región segmentada
    segmented_region = closing

    return segmented_region

def paste_mask(img_original, img_mask):
    alpha = 1
    img_plus_mask = cv2.addWeighted(img_original, 1, img_mask, alpha, 0)
    return img_plus_mask

if __name__ == "__main__":

    input_dir_png = "Estudio_1_png"

    # Cargar la imagen
    imagen = cv2.imread(input_dir_png + '/BOULLOSA-MADRAZO-ANTONIO.transformed150.png',
                        cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises

    # Definir una máscara de suavizado
    kernel = (1 / (11 ** 2)) * np.ones((11, 11), np.uint8)
    # kernel = cv2.getGaussianKernel(11,1)

    umbral = 157

    erosion, dilatacion = erotion_dilation(imagen, kernel, umbral)

    open_close = opening_closing(imagen, umbral, 7)

    imagen_con_erosion = paste_mask(imagen,erosion)
    imagen_con_cierre = paste_mask(imagen,open_close)

    # Guardar imagen
    cv2.imwrite("Imagen original.png", imagen)
    cv2.imwrite("Region de interes.png", open_close)
    cv2.imwrite("Imagen con cierre.png", imagen_con_cierre)

    # MOstrar imágenes
    cv2.imshow("Imagen original", imagen)
    #cv2.imshow("Erosion", erosion)
    #cv2.imshow("Dilatacion", dilatacion)
    #cv2.imshow("Imagen con erosion",imagen_con_erosion)
    cv2.imshow("Imagen con cierre", imagen_con_cierre)

    cv2.waitKey(0)
    cv2.destroyAllWindows()