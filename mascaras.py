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

def detectar_contornos_y_areas(image):
    # Detectar contornos
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una copia de la imagen para visualizar los contornos
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Iterar sobre los contornos y calcular las áreas
    areas = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        areas.append((i, area))

        # Dibujar el contorno en la imagen
        color = (0, 255, 0)  # Verde
        cv2.drawContours(output_image, [contour], -1, color, thickness=2)

        # Obtener el punto central para etiquetar el área
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Escribir el área cerca del contorno
        cv2.putText(output_image, f"{area:.0f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Ordenar las áreas de mayor a menor
    areas.sort(key=lambda x: x[1], reverse=True)

    # Imprimir las áreas
    #print("Áreas de los contornos detectados:")
    #for idx, area in areas:
    #    print(f"Contorno {idx}: Área = {area}")

    return output_image, areas

def encontrar_contorno_por_area(img, target_area_min, target_area_max, tolerance=50):
    """
    Función que busca un contorno en una imagen cuya área esté cerca del área objetivo.

    Parámetros:
        - image_path (str): Ruta de la imagen en escala de grises.
        - target_area (int): El área deseada para el contorno.
        - tolerance (int, opcional): Tolerancia para considerar un área similar. El valor predeterminado es 50.

    Devuelve:
        - result (ndarray): Imagen resultante con el contorno seleccionado.
        - selected_contour (ndarray o None): El contorno seleccionado o None si no se encontró.
    """

    # Detectar contornos
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara vacía para dibujar el contorno seleccionado
    mask = np.zeros_like(img)

    # Variable para almacenar el contorno encontrado
    selected_contour = None

    # Iterar sobre los contornos y buscar el que tenga un área cercana al área deseada

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= target_area_min - tolerance and area <=target_area_max + tolerance:
            #if abs(area - target_area) <= tolerance:
            selected_contour = contour
            break

    # Si se encuentra un contorno, dibujarlo en la máscara
    if selected_contour is not None:
        cv2.drawContours(mask, [selected_contour], -1, 255, thickness=cv2.FILLED)
        #print(f"Contorno con área cercana a {target_area} encontrado. Área: {cv2.contourArea(selected_contour)}")
    #else:
    #    print(f"No se encontró ningún contorno con un área cercana a {target_area}.")


    # Aplicar la máscara para conservar solo la región del contorno de interés
    region_of_interest = cv2.bitwise_and(img, mask)

    return region_of_interest, selected_contour

if __name__ == "__main__":

    kernel_size = 7

    # Definir una máscara de suavizado
    kernel = (1 / (kernel_size ** 2)) * np.ones((kernel_size, kernel_size), np.uint8)
    # kernel = cv2.getGaussianKernel(11,1)

    umbral = 155

    input_dir_png = "Estudio_1_png"
    output_dir_background = "Estudio_1_png/background_remove"
    output_dir_mask = "Estudio_1_png/mascaras"
    output_dir_area = "Estudio_1_png/area_principal"
    output_dir_segmento = "Estudio_1_png/segmentacion"
    output_dir_final = "Estudio_1_png/resultado_final"

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

            # Área principal
            img_areas, areas = detectar_contornos_y_areas(img_section)
            # Guardar la imagen en formato PNG
            output_filename_area = os.path.splitext(file)[0] + '_area.png'
            output_filepath_area = os.path.join(output_dir_area, output_filename_area)
            cv2.imwrite(output_filepath_area, img_areas)

            # Área principal
            cierre = opening_closing(img, umbral, kernel_size)
            segment, selected_contour = encontrar_contorno_por_area(img_section, 6500, 9500, 50 )
            # Guardar la imagen en formato PNG
            output_filename_segmento = os.path.splitext(file)[0] + '_segmentado.png'
            output_filepath_segmento = os.path.join(output_dir_segmento, output_filename_segmento)
            cv2.imwrite(output_filepath_segmento, segment)

            # Área principal
            img_con_mascara_area = paste_mask(img, segment)
            # Guardar la imagen en formato PNG
            output_filename_final = os.path.splitext(file)[0] + '_imagen_segmentada.png'
            output_filepath_final = os.path.join(output_dir_final, output_filename_final)
            cv2.imwrite(output_filepath_final, img_con_mascara_area)

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