import cv2
import numpy as np



input_dir_png = "Estudio_1_png"

# Cargar la imagen
imagen = cv2.imread(input_dir_png+'/BOULLOSA-MADRAZO-ANTONIO.transformed150.png', cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises

# Definir una máscara de suavizado
kernel = (1/(11**2))*np.ones((11,11), np.uint8)
#kernel = cv2.getGaussianKernel(11,1)
print(kernel)

# IMagen blur
imagen_filtrada = cv2.filter2D(imagen,-1,kernel)

# Cobertir a binario
_, imagen_binaria = cv2.threshold(imagen_filtrada, 145, 255, cv2.THRESH_BINARY)

# Aplicar erosion
erosion = cv2.erode(imagen_binaria, kernel, iterations=1)

# Aplicar dilatación
dilatacion = cv2.dilate(imagen_binaria, kernel, iterations=1)


# Guardar imagen
cv2.imwrite("Imagen original.png",imagen)
cv2.imwrite("Imagen suavizada.png",imagen_filtrada)
cv2.imwrite("Imagen binaria.png",imagen_binaria)
cv2.imwrite("Erosion.png",erosion)
cv2.imwrite("Dilatacion.png",dilatacion)


# MOstrar imágenes
cv2.imshow("Imagen original", imagen)
cv2.imshow("Imagen suavizada", imagen_filtrada)
cv2.imshow("Imagen binaria", imagen_binaria)
cv2.imshow("Erosion",erosion)
cv2.imshow("Dilatacion",dilatacion)

cv2.waitKey(0)
cv2.destroyAllWindows()
