import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import json
import os

# Leer y preparar datos
def load_data(image_dir, json_dir):
    images = []
    masks = []
    for json_file in os.listdir(json_dir):
        with open(os.path.join(json_dir, json_file), 'r') as f:
            annotations = json.load(f)
        mask = np.zeros((height, width), dtype=np.uint8)  # Cambia a las dimensiones correctas
        for obj in annotations["objects"]:
            points = np.array(obj["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)  # Crear la máscara binaria
        img = cv2.imread(os.path.join(image_dir, json_file.replace(".json", ".png")), cv2.IMREAD_GRAYSCALE)
        images.append(img)
        masks.append(mask)
    return np.array(images), np.array(masks)

# U-Net model
def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return models.Model(inputs, outputs)

if __name__ == "__main__":

    # Entrenamiento
    image_dir = "path/to/images"
    json_dir = "path/to/jsons"
    images, masks = load_data(image_dir, json_dir)

    images = images / 255.0  # Normalizar
    masks = masks / 255.0    # Normalizar

    model = build_unet((height, width, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(images, masks, epochs=10, batch_size=16)

    # Predicción
    preds = model.predict(images)
    for i, pred in enumerate(preds):
        pred_binary = (pred > 0.5).astype(np.uint8)  # Umbral
        contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(images[i], [contour], -1, (255, 0, 0), 2)  # Dibujar contornos
