#Compare DeepGaze III with OpenCV
#Highlight red and yellow regions
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import deepgaze_pytorch
from datetime import datetime

# ----- CONFIG -----
IMAGE_PATH = "BMW_CR09799707874329362433_v2_image.png"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ----- LOAD IMAGE -----
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
resized_rgb = cv2.resize(image_rgb, (1024, 768))  # (W, H)

# ----- OpenCV STATIC SALIENCY -----
opencv_saliency = cv2.saliency.StaticSaliencyFineGrained_create()
_, saliency_map = opencv_saliency.computeSaliency(resized_rgb)
saliency_map_opencv = (saliency_map * 255).astype("uint8")

# ----- DEEPGAZE III -----
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((768, 1024)),
    transforms.ToTensor(),
])

image_pil = Image.fromarray(resized_rgb)
input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
centerbias = torch.zeros((1, 768, 1024)).to(DEVICE)
width, height = 1024, 768
x_hist = torch.tensor([[width // 2, width // 4, 3 * width // 4, width // 3]]).float().to(DEVICE)
y_hist = torch.tensor([[height // 2, height // 4, 3 * height // 4, height // 3]]).float().to(DEVICE)

with torch.no_grad():
    prediction = model(input_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)
    prediction = prediction.squeeze().cpu().numpy()

deepgaze_map = (prediction - prediction.min()) / (prediction.max() - prediction.min())


# ----- ATTENTION ZONE VISUALIZATION -----
def highlight_attention(image, saliency_map, percentile=85):
    """Highlight only top attention zones in red/yellow"""
    # Normalize and threshold
    norm_map = (saliency_map * 255).astype("uint8")
    threshold = np.percentile(norm_map, percentile)

    # Create heatmap (red = top 5%, yellow = next 10%)
    heatmap = np.zeros((*saliency_map.shape, 3))

    # Yellow zones (85th-95th percentile)
    yellow_mask = (norm_map > threshold) & (norm_map < threshold * 1.15)
    heatmap[yellow_mask] = [255, 255, 0]  # Yellow

    # Red zones (top 5%)
    red_mask = norm_map >= threshold * 1.15
    heatmap[red_mask] = [255, 0, 0]  # Red

    # Blend with original image
    blended = cv2.addWeighted(image, 0.7, heatmap.astype('uint8'), 0.3, 0)
    return blended


# Process both saliency methods
opencv_highlight = highlight_attention(resized_rgb, saliency_map)
deepgaze_highlight = highlight_attention(resized_rgb, deepgaze_map)

# ----- VISUALIZATION -----
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 1])

# Original Image
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(resized_rgb)
ax0.set_title("Original Advertisement")
ax0.axis("off")

# OpenCV Results
ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(opencv_highlight)
ax1.set_title("OpenCV High Attention Zones\n(Red = Top 5%, Yellow = Next 10%)")
ax1.axis("off")

# DeepGaze Results
ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(deepgaze_highlight)
ax2.set_title("DeepGaze III High Attention Zones\n(Red = Top 5%, Yellow = Next 10%)")
ax2.axis("off")

# Comparison
ax3 = fig.add_subplot(gs[1, 1])
ax3.imshow(resized_rgb)
ax3.imshow(deepgaze_map, cmap='hot', alpha=0.5)
ax3.set_title("Full DeepGaze Heatmap (Reference)")
ax3.axis("off")

# Interpretation
interpretation = """INTERPRETATION GUIDE:
• RED ZONES = Highest attention (first 5% fixation probability) - Place key messages here
• YELLOW ZONES = Secondary attention (next 10% probability) - Secondary elements work here
• OpenCV shows low-level visual attention (contrast/edges)
• DeepGaze predicts human gaze patterns including cognitive factors
• Compare both to see what grabs immediate vs considered attention"""

text_box = fig.add_subplot(gs[2, :])
text_box.text(0.05, 0.5, interpretation,
              ha='left', va='center',
              wrap=True, fontsize=12,
              bbox=dict(facecolor='whitesmoke', alpha=0.8))
text_box.axis('off')

plt.tight_layout()
output_filename = f"attention_analysis_{TIMESTAMP}.png"
plt.savefig(output_filename, dpi=200, bbox_inches='tight')
plt.show()

print(f"Analysis saved to: {output_filename}")