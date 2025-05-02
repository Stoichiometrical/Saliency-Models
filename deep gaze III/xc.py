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
IMAGE_PATH = "BMW_CR09799707874329362433_v2_image.png"  # Replace with your file path
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")  # For unique filenames

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
# Historical gaze positions (simulated)
width, height = 1024, 768
x_hist = torch.tensor([[width // 2, width // 4, 3 * width // 4, width // 3]]).float().to(DEVICE)
y_hist = torch.tensor([[height // 2, height // 4, 3 * height // 4, height // 3]]).float().to(DEVICE)

with torch.no_grad():
    prediction = model(input_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)
    prediction = prediction.squeeze().cpu().numpy()

deepgaze_map = (prediction - prediction.min()) / (prediction.max() - prediction.min())

# ----- VISUALIZATION -----
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[4, 1])

# Image plots
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(resized_rgb)
ax1.set_title("Original Ad")
ax1.axis("off")

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(resized_rgb)
ax2.imshow(saliency_map_opencv, cmap='hot', alpha=0.6)
ax2.set_title("OpenCV Static Saliency")
ax2.axis("off")

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(resized_rgb)
ax3.imshow(deepgaze_map, cmap='hot', alpha=0.6)
ax3.set_title("DeepGaze III Saliency")
ax3.axis("off")

# Interpretation text
interpretation = """INTERPRETATION GUIDE:
1. HEATMAP COLORS: Red/Yellow = High attention areas, Blue = Low attention areas
2. OPENCV SALIENCY: Highlights basic visual features (contrast, edges, color)
3. DEEPGAZE III: Predicts human eye movements including cognitive factors
4. MARKETING INSIGHTS:
   - Place key messages in hot spots (red/yellow)
   -Maximum attention (peaks) - red, High attention (but slightly less than red) -yellow
   - Avoid critical info in cool areas (blue)
   -Primary message should be in red, secondary should be in yellow
   - Compare both maps to understand automatic vs cognitive attention"""

text_box = fig.add_subplot(gs[1, :])
text_box.text(0.05, 0.5, interpretation,
             ha='left', va='center',
             wrap=True, fontsize=12)
text_box.axis('off')

plt.tight_layout()
output_filename = f"saliency_comparison_{TIMESTAMP}.png"
plt.savefig(output_filename, dpi=200, bbox_inches='tight')
plt.show()

print(f"Results saved as: {output_filename}")

# import os
# import cv2
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms
# import deepgaze_pytorch
#
# # ----- CONFIG -----
# IMAGE_PATH = "BMW_CR09799707874329362433_v2_image.png"  # Replace with your file path
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# # ----- LOAD IMAGE -----
# image_bgr = cv2.imread(IMAGE_PATH)
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# resized_rgb = cv2.resize(image_rgb, (1024, 768))  # (W, H)
#
# # ----- OpenCV STATIC SALIENCY -----
# opencv_saliency = cv2.saliency.StaticSaliencyFineGrained_create()
# _, saliency_map = opencv_saliency.computeSaliency(resized_rgb)
# saliency_map_opencv = (saliency_map * 255).astype("uint8")
#
# # ----- DEEPGAZE III -----
# model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)
# model.eval()
#
# transform = transforms.Compose([
#     transforms.Resize((768, 1024)),
#     transforms.ToTensor(),
# ])
#
# image_pil = Image.fromarray(resized_rgb)
# input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
# centerbias = torch.zeros((1, 768, 1024)).to(DEVICE)
# # Historical gaze positions (simulated)
# width, height = 1024, 768
# x_hist = torch.tensor([[width // 2, width // 4, 3 * width // 4, width // 3]]).float().to(DEVICE)
# y_hist = torch.tensor([[height // 2, height // 4, 3 * height // 4, height // 3]]).float().to(DEVICE)
#
# with torch.no_grad():
#     prediction = model(input_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)
#     prediction = prediction.squeeze().cpu().numpy()
#
# deepgaze_map = (prediction - prediction.min()) / (prediction.max() - prediction.min())
#
# # ----- VISUALIZATION -----
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#
# axs[0].imshow(resized_rgb)
# axs[0].set_title("Original Ad")
# axs[0].axis("off")
#
# axs[1].imshow(resized_rgb)
# axs[1].imshow(saliency_map_opencv, cmap='hot', alpha=0.6)
# axs[1].set_title("OpenCV Static Saliency")
# axs[1].axis("off")
#
# axs[2].imshow(resized_rgb)
# axs[2].imshow(deepgaze_map, cmap='hot', alpha=0.6)
# axs[2].set_title("DeepGaze III Saliency")
# axs[2].axis("off")
#
# plt.tight_layout()
# plt.savefig("saliency_comparison.png", dpi=200)
# plt.show()
