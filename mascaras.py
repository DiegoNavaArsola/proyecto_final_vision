import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_background(img, output_path):

    imagen_filtrada = cv2.bilateralFilter(img, d=25, sigmaColor=100, sigmaSpace=100)

    # Aplicar un umbral adaptativo para separar fondo y objeto
    _, binary = cv2.threshold(imagen_filtrada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invertir la máscara binaria
    #binary_inv = cv2.bitwise_not(binary)
    binary_inv = binary

    # Aplicar operaciones morfológicas para eliminar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)

    # Aplicar la máscara limpia sobre la imagen original
    foreground = cv2.bitwise_and(img, img, mask=binary_cleaned)


    return foreground

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

    kernel_size = 7

    # Definir una máscara de suavizado
    kernel = (1 / (kernel_size ** 2)) * np.ones((kernel_size, kernel_size), np.uint8)
    # kernel = cv2.getGaussianKernel(11,1)

    umbral = 155

    input_dir_png = "Estudio_1_png"
    output_dir_background = "Estudio_1_png/background_remove"
    output_dir_mask = "Estudio_1_png/mascaras"

    for file in os.listdir(input_dir_png):
        if file.lower().endswith('.png'):

            # Cargar la imagen
            img = cv2.imread(input_dir_png+"/"+file, cv2.IMREAD_GRAYSCALE)

            # Quitar fondo
            img_foreground = remove_background(img,None)

            output_filename_bck = os.path.splitext(file)[0] + '_foreground.png'
            output_filepath_bck = os.path.join(output_dir_background, output_filename_bck)
            cv2.imwrite(output_filepath_bck, img_foreground)

            # Sección de interés
            img_section = opening_closing(img_foreground,umbral,kernel_size)

            img_mask = paste_mask(img_foreground, img_section)

            # Guardar la imagen en formato PNG
            output_filename = os.path.splitext(file)[0] + '_region.png'
            output_filepath = os.path.join(output_dir_mask, output_filename)
            cv2.imwrite(output_filepath,img_mask)
            print(f"Imagen {output_filename} -> Correcto")


    """imagen = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

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
    cv2.destroyAllWindows()"""