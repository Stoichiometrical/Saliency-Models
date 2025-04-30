import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import deepgaze_pytorch
from scipy.ndimage import gaussian_filter

# Initialize both models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
deepgaze_model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
deepgaze_model.eval()
opencv_saliency = cv2.saliency.StaticSaliencyFineGrained_create()


# Hybrid processing function
def analyze_ad_hybrid(image_path):
    # ===== OpenCV Processing =====
    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # OpenCV saliency
    _, opencv_heatmap = opencv_saliency.computeSaliency(image_cv)
    opencv_heatmap = (opencv_heatmap * 255).astype("uint8")
    opencv_heatmap = cv2.GaussianBlur(opencv_heatmap, (25, 25), 0)

    # ===== DeepGaze Processing =====
    image_pil = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((768, 1024)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    centerbias = torch.zeros((1, 768, 1024)).to(device)

    # Synthetic fixations
    width, height = 1024, 768
    x_hist = torch.tensor([[width // 2, width // 4, 3 * width // 4, width // 3]]).float().to(device)
    y_hist = torch.tensor([[height // 2, height // 4, 3 * height // 4, height // 3]]).float().to(device)

    with torch.no_grad():
        deepgaze_output = deepgaze_model(image_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)

    deepgaze_heatmap = deepgaze_output.squeeze().cpu().numpy()
    deepgaze_heatmap = gaussian_filter(deepgaze_heatmap, sigma=5)

    # ===== Fusion =====
    # Resize OpenCV heatmap to match DeepGaze
    opencv_resized = cv2.resize(opencv_heatmap, (1024, 768)) / 255.0

    # Combine heatmaps (weighted average)
    combined_heatmap = 0.7 * deepgaze_heatmap + 0.3 * opencv_resized

    # ===== Metrics =====
    metrics = {
        "Clarity": np.max(combined_heatmap),
        "Focus": np.std(combined_heatmap),
        "Engagement": np.mean(combined_heatmap),
        "DeepGaze Dominance": np.mean(deepgaze_heatmap > opencv_resized)
    }

    return {
        "original": image_rgb,
        "opencv_heatmap": opencv_heatmap,
        "deepgaze_heatmap": (deepgaze_heatmap * 255).astype(np.uint8),
        "combined_heatmap": (combined_heatmap * 255).astype(np.uint8),
        "metrics": metrics
    }


# ===== Visualization =====
def plot_hybrid_results(results):
    plt.figure(figsize=(20, 10))
    for idx, (name, data) in enumerate(results.items()):
        # Original Image
        plt.subplot(3, 4, idx * 4 + 1)
        plt.imshow(data['original'])
        plt.title(f"{name}\nOriginal", fontsize=10)
        plt.axis('off')

        # OpenCV Heatmap
        plt.subplot(3, 4, idx * 4 + 2)
        plt.imshow(data['opencv_heatmap'], cmap='hot')
        plt.title("OpenCV Saliency", fontsize=10)
        plt.axis('off')

        # DeepGaze Heatmap
        plt.subplot(3, 4, idx * 4 + 3)
        plt.imshow(data['deepgaze_heatmap'], cmap='hot')
        plt.title("DeepGaze III", fontsize=10)
        plt.axis('off')

        # Combined Heatmap
        plt.subplot(3, 4, idx * 4 + 4)
        plt.imshow(data['combined_heatmap'], cmap='hot')

        # Overlay metrics
        metrics_text = "\n".join([f"{k}: {v:.2f}" for k, v in data['metrics'].items()])
        plt.text(10, 30, metrics_text,
                 color='white', fontsize=9,
                 bbox=dict(facecolor='black', alpha=0.7))

        plt.title("Combined Analysis", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# ===== Execution =====
ad_paths = {
    "Our_Ad": "ads/Audi_CR08545138718757879809_v1_image.png",
    "Competitor_Ad": "ads/BMW_CR09799707874329362433_v2_image.png"
}

results = {name: analyze_ad_hybrid(path) for name, path in ad_paths.items()}
plot_hybrid_results(results)