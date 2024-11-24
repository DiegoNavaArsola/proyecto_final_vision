import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Bloques de codificación
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Bloques de decodificación
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)

        # Salida
        self.final = nn.Conv2d(64, 1, kernel_size=1, activation=None)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Codificación
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decodificación
        d4 = self.dec4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d1 = self.dec1(d2)
        d1 = torch.cat((d1, e1), dim=1)

        # Salida
        return torch.sigmoid(self.final(d1))

class MedicalDatasetFromJSON(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        json_path = os.path.join(self.json_dir, img_name.replace(".png", ".json"))

        # Leer imagen
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Crear máscara desde el archivo JSON
        mask = np.zeros_like(image, dtype=np.uint8)
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        for obj in annotations["objects"]:
            points = np.array(obj["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)  # Rellenar las áreas etiquetadas

        # Redimensionar para la U-Net
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # Normalizar
        image = image / 255.0
        mask = mask / 255.0

        # Convertir a tensores
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)    # (1, H, W)

        return image, mask


if __name__ == "__main__":

    # Directorios de datos
    image_dir = "Imagenes_entrenamiento"
    json_dir = "Etiquetas"

    # Crear dataset y dataloader
    dataset = MedicalDatasetFromJSON(image_dir, json_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


    # Inicializar modelo, pérdida y optimizador
    model = UNet().cuda()
    criterion = nn.BCELoss()  # Binary Cross-Entropy
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Entrenamiento
    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images, masks = images.cuda(), masks.cuda()

            # Forward
            preds = model(images)
            loss = criterion(preds, masks)
            epoch_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.cuda()
            preds = model(images)
            preds = (preds > 0.5).float()  # Umbral binario

            for i, pred in enumerate(preds):
                pred_np = pred.squeeze().cpu().numpy()

                # Detección de contornos
                contours, _ = cv2.findContours(pred_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                original_image = (images[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)

                plt.imshow(original_image)
                plt.show()
