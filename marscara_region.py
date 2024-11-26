import cv2
import numpy as np
import matplotlib.pyplot as plt

# Crear una imagen de ejemplo y una máscara
def create_example_image_and_mask():
    # Imagen base (un gradiente de color)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    for i in range(image.shape[1]):
        image[:, i, :] = [i % 256, (i * 2) % 256, (i * 3) % 256]

    # Máscara inicial (un círculo blanco sobre fondo negro)
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(mask, (150, 100), 50, 255, -1)  # Círculo en el centro

    return image, mask

# Aplicar la máscara a la imagen para mostrar solo la región seleccionada
def apply_mask_to_image(image, mask):
    # Aplicar la máscara directamente a la imagen
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# Crear la imagen y la máscara
image, mask = create_example_image_and_mask()

# Generar la imagen con la máscara aplicada
masked_gradient = apply_mask_to_image(image, mask)

# Visualización de los resultados
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Imagen Original")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Máscara Original")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(masked_gradient, cv2.COLOR_BGR2RGB))
axes[2].set_title("Resultado Enmascarado")
axes[2].axis("off")

plt.tight_layout()
plt.show()