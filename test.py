from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load and resize the image
image_path = "Audi_CR08545138718757879809_v1_image.png"
image = Image.open(image_path).convert('RGB')
image_resized = image.resize((1024, 768))
image_np = np.array(image_resized)

# Use OpenCV's saliency detector (simulate attention heatmap)
opencv_saliency = cv2.saliency.StaticSaliencyFineGrained_create()
success, saliency_map = opencv_saliency.computeSaliency(image_np)
saliency_map = (saliency_map * 255).astype("uint8")
saliency_map_colored = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
saliency_map_colored = cv2.cvtColor(saliency_map_colored, cv2.COLOR_BGR2RGB)

# Blend original image with heatmap
blended = cv2.addWeighted(image_np, 0.5, saliency_map_colored, 0.5, 0)

# Display result
plt.figure(figsize=(12, 6))
plt.imshow(blended)
plt.axis('off')
plt.title("Simulated Attention Heatmap Overlay")
plt.tight_layout()
plt.show()
