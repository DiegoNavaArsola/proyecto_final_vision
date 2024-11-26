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

    #return foreground
    return cv2.equalizeHist(foreground)

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

def detect_areas(image):
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

    return output_image, areas

def find_contopurs_per_area(img, target_area_min, target_area_max, tolerance=50):

    # Estatus de busqueda por área
    status = False

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
            status_area = True
            #if abs(area - target_area) <= tolerance:
            selected_contour = contour
            break

    # Si se encuentra un contorno, dibujarlo en la máscara
    if selected_contour is not None:
        cv2.drawContours(mask, [selected_contour], -1, 255, thickness=cv2.FILLED)

    # Aplicar la máscara para conservar solo la región del contorno de interés
    region_of_interest = cv2.bitwise_and(img, mask)

    return region_of_interest, selected_contour, status

def detect_similar_shapes(image, mask, similarity_threshold):
    """
       Detecta regiones en la imagen cuyo contorno sea similar al contorno de la máscara.

       Args:
       - image: Imagen binarizada de entrada (NumPy array, escala de grises).
       - mask: Máscara binarizada (NumPy array, escala de grises).
       - similarity_threshold: Umbral de similitud para determinar coincidencias (valor bajo = más estricto).
       """

    status_form = False

    # Verificar que la máscara tenga contornos
    base_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not base_contours:
        print("No se encontraron contornos en la máscara base.")
        return
    base_contour = base_contours[0]  # Asumimos que la máscara tiene un solo contorno principal

    # Encontrar contornos en la imagen binarizada
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una copia de la imagen original (en color) para visualizar los resultados
    image_result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Iterar sobre los contornos encontrados
    for contour in contours:
        # Calcular la similitud entre el contorno actual y el de la máscara base
        similarity = cv2.matchShapes(base_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)

        # Verificar si el contorno es lo suficientemente similar
        if similarity < similarity_threshold:
            status_form = True
            # Dibujar el contorno en la imagen
            cv2.drawContours(image_result, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(image_result, f"Similar ({similarity:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)
        else:
            status_form = False
            break
    return image, image_result, status_form

def save_image_dir(img,file_name,suflix, format, output_dir):
    suflix_p_format = suflix + format
    output_filename = os.path.splitext(file_name)[0] + suflix_p_format
    output_filepath = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_filepath, img)

if __name__ == "__main__":

    kernel_size = 7

    umbral = 200
    min_area = 5300
    max_area = 9300

    mascara_patron = cv2.imread("Estudio_1_png/segmentacion/BOULLOSA-MADRAZO-ANTONIO.transformed147_segmentado.png", cv2.IMREAD_GRAYSCALE)
    umbral_similitud = 0.04


    input_dir_png = "Estudio_1_png"
    output_dir_background = "Estudio_1_png/background_remove"
    output_dir_mask = "Estudio_1_png/mascaras"
    output_dir_area = "Estudio_1_png/area_principal"
    output_dir_segmento = "Estudio_1_png/segmentacion"
    output_dir_similar = "Estudio_1_png/forma_similar"
    output_dir_final = "Estudio_1_png/resultado_final"

    for file in os.listdir(input_dir_png):
        if file.lower().endswith('.png'):

            # Cargar la imagen
            img = cv2.imread(input_dir_png+"/"+file, cv2.IMREAD_GRAYSCALE)

            # Quitar fondo
            img_foreground = remove_background(img,None)
            save_image_dir(img_foreground, file, "_foreground", ".png", output_dir_background)

            # Cierre para encontrár área de interés segun el umbral
            img_section = opening_closing(img_foreground,umbral,kernel_size)
            img_mask = paste_mask(img_foreground, img_section)
            save_image_dir(img_mask, file, "_region", ".png", output_dir_mask)

            # Selección de contornos y cálculos de area
            img_areas, areas = detect_areas(img_section)
            save_image_dir(img_areas, file, "_area", ".png", output_dir_area)

            # Área principal
            segment, selected_contour, status_area = find_contopurs_per_area(img_section, min_area, max_area, 50)
            save_image_dir(segment, file, "_segmentado", ".png", output_dir_segmento)

            # Comparación de forma
            _, mask_borders, status_form = detect_similar_shapes(segment, mascara_patron, umbral_similitud)
            if status_form:
                save_image_dir(mask_borders,file,"_forma_similar",".png",output_dir_similar)

            # Máscara de área sobre imagen
            img_con_mascara_area = paste_mask(img, segment)
            save_image_dir(img_con_mascara_area,file,'_imagen_segmentada',".png",output_dir_final)


            print(f"Imagen {file} -> Correcto")
