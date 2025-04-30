import torch
from pysaliency.external_models import DeepGazeIIE
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Initialize DeepGaze IIE model
model = DeepGazeIIE(centerbias = torch.zeros((1, 768, 1024)).to(device))



def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return img, transform(img).unsqueeze(0)


def generate_saliency(model, img_tensor, original_size):
    with torch.no_grad():
        saliency = model.saliency_map(img_tensor)
    return F.interpolate(saliency, size=original_size[::-1], mode='bilinear').squeeze().numpy()


def visualize(image, saliency_map, alpha=0.5):
    plt.figure(figsize=(15, 5))

    # Normalize saliency
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(saliency_map, cmap='magma')
    plt.title("Saliency Heatmap")

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(saliency_map, cmap='magma', alpha=alpha)
    plt.title("Overlay")

    plt.tight_layout()
    plt.show()


# Usage
image_path = "Audi_CR08545138718757879809_v2_image.png"
img, img_tensor = process_image(image_path)

saliency_map = generate_saliency(model, img_tensor, img.size)
visualize(img, saliency_map)