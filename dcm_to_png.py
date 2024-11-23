import os
import pydicom
import numpy as np
from PIL import Image


def convert_dcm_to_png(input_dir, output_dir):
    """
    Convierte todas las imágenes DICOM (.dcm) de un directorio a formato PNG.

    Args:
        input_dir (str): Directorio donde están los archivos DICOM.
        output_dir (str): Directorio donde se guardarán las imágenes PNG.
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Recorrer todos los archivos del directorio de entrada
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.dcm'):  # Verifica que el archivo sea .dcm
            filepath = os.path.join(input_dir, filename)
            try:
                # Leer el archivo DICOM
                dicom = pydicom.dcmread(filepath)
                # Convertir los datos del DICOM a un array de numpy
                pixel_array = dicom.pixel_array

                # Normalizar los valores a un rango de 0-255
                normalized_array = ((pixel_array - np.min(pixel_array)) /
                                    (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)

                # Crear una imagen con PIL
                img = Image.fromarray(normalized_array)

                # Guardar la imagen en formato PNG
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_filepath = os.path.join(output_dir, output_filename)
                img.save(output_filepath)

                print(f"Convertido: {filename} -> {output_filename}")
            except Exception as e:
                print(f"Error al convertir {filename}: {e}")

if __name__ == "__main__":


    input_e1 = [f"Estudio 1/Tiempo {i+1}" for i in range(0,9)]
    print(input_e1)

    for i in input_e1:
        # Rutas de entrada y salida
        input_directory = i
        output_directory = "Estudio_1_png"

        # Llamar a la función
        convert_dcm_to_png(input_directory, output_directory)