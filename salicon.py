from pysaliency.external_datasets import get_SALICON
from pysaliency.external_models import SaliconModel
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 1. Load SALICON model
salicon_model = SaliconModel()


# 2. Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((480, 640)),  # SALICON's native resolution
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


# 3. Generate saliency map
def generate_saliency(model, img_tensor, original_size):
    saliency = model.saliency_map(img_tensor.numpy())  # SALICON expects numpy array
    saliency = Image.fromarray(saliency[0]).resize(original_size, Image.BILINEAR)
    return np.array(saliency)


# 4. Visualization
def visualize(image, saliency_map, alpha=0.5):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(saliency_map, cmap='hot')
    plt.title("Saliency Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(saliency_map, cmap='hot', alpha=alpha)
    plt.title(f"Overlay (α={alpha})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Execution
image_path = "your_image.jpg"
original_img, img_tensor = preprocess_image(image_path)
saliency_map = generate_saliency(salicon_model, img_tensor, original_img.size)
visualize(original_img, saliency_map)