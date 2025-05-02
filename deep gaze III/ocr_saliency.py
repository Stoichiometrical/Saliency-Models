import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import easyocr
from torchvision import transforms
import deepgaze_pytorch
from datetime import datetime
import warnings

# CONFIG
IMAGE_PATH = "Audi_CR08545138718757879809_v1_image.png"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CTA_KEYWORDS = ["learn more", "book a test drive", "shop now", "order", "sign up", "try now", "explore", "test drive"]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load image
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
resized_rgb = cv2.resize(image_rgb, (1024, 768))  # W x H


# ---------- OpenCV Saliency (Robust Implementation) ----------
def compute_opencv_saliency(image):
    """Handle all possible OpenCV saliency method scenarios"""
    saliency_map = None
    method_used = "None"

    try:
        # Try Spectral Residual (most common)
        if hasattr(cv2, 'saliency') and 'StaticSaliencySpectralResidual_create' in dir(cv2.saliency):
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(image)
            if success:
                method_used = "SpectralResidual"
                return saliency_map, method_used

        # Try Fine Grained (older versions)
        if hasattr(cv2, 'saliency') and 'StaticSaliencyFineGrained_create' in dir(cv2.saliency):
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            success, saliency_map = saliency.computeSaliency(image)
            if success:
                method_used = "FineGrained"
                return saliency_map, method_used

        # Try Objectness as last resort
        if hasattr(cv2, 'saliency') and 'ObjectnessBING_create' in dir(cv2.saliency):
            saliency = cv2.saliency.ObjectnessBING_create()
            success, saliency_map = saliency.computeSaliency(image)
            if success:
                method_used = "Objectness"
                return saliency_map, method_used

    except Exception as e:
        warnings.warn(f"Saliency computation failed: {str(e)}")

    # Fallback: Create center-weighted dummy map
    h, w = image.shape[:2]
    saliency_map = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    cv2.circle(saliency_map, (cx, cy), min(w, h) // 3, 1.0, -1)
    method_used = "DummyFallback"

    return saliency_map, method_used


saliency_map, saliency_method = compute_opencv_saliency(resized_rgb)
print(f"Used saliency method: {saliency_method}")
saliency_map_opencv = (saliency_map * 255).astype("uint8")

# ---------- DeepGaze III (Updated Implementation) ----------
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

# ---------- EasyOCR CTA Detection ----------
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())


def detect_cta_easyocr(image_rgb):
    results = reader.readtext(image_rgb)
    boxes = []
    for (bbox, text, conf) in results:
        word = text.strip().lower()
        if any(keyword in word for keyword in CTA_KEYWORDS):
            x1, y1 = map(int, bbox[0])
            x2, y2 = map(int, bbox[2])
            boxes.append((x1, y1, x2, y2, word, conf))
    return boxes


cta_boxes = detect_cta_easyocr(resized_rgb)


# ---------- Visualization Functions ----------
def draw_cta_boxes(image, boxes, color=(0, 255, 0)):
    img_copy = image.copy()
    for (x1, y1, x2, y2, word, conf) in boxes:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        label = f"{word} ({conf:.2f})"
        cv2.putText(img_copy, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_copy


def highlight_attention(image, saliency_map, cta_boxes=None, alpha=0.6):
    """Highlight attention zones with red/yellow colormap and CTA boxes"""
    norm_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    heatmap = cv2.applyColorMap((norm_map * 255).astype('uint8'), cv2.COLORMAP_HOT)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)

    # Draw CTA boxes if provided
    if cta_boxes:
        for (x1, y1, x2, y2, word, conf) in cta_boxes:
            # Draw thicker green rectangle
            cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Add text label
            label = f"{word} ({conf:.2f})"
            cv2.putText(blended, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return blended


# Create visualizations - now passing cta_boxes to highlight_attention
image_with_cta = draw_cta_boxes(resized_rgb, cta_boxes)
opencv_attention = highlight_attention(resized_rgb, saliency_map, cta_boxes)
deepgaze_attention = highlight_attention(resized_rgb, deepgaze_map, cta_boxes)


# # Create visualizations
# image_with_cta = draw_cta_boxes(resized_rgb, cta_boxes)
# opencv_attention = highlight_attention(resized_rgb, saliency_map)
# deepgaze_attention = highlight_attention(resized_rgb, deepgaze_map)

# Add CTA boxes to attention maps
for box in cta_boxes:
    x1, y1, x2, y2, word, conf = box
    cv2.rectangle(opencv_attention, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(deepgaze_attention, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ---------- Enhanced Visualization ----------
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

# Original with CTAs
axs[0, 0].imshow(image_with_cta)
axs[0, 0].set_title("1. CTA Detection (EasyOCR)")
axs[0, 0].axis("off")

# OpenCV Saliency
axs[0, 1].imshow(opencv_attention)
axs[0, 1].set_title(f"2. OpenCV Saliency ({saliency_method})")
axs[0, 1].axis("off")

# DeepGaze III
axs[1, 0].imshow(deepgaze_attention)
axs[1, 0].set_title("3. DeepGaze III with CTA")
axs[1, 0].axis("off")

# Interpretation
interpretation = f"""INTERPRETATION GUIDE:
• GREEN BOXES: Detected Call-to-Action (CTA) elements
• SALIENCY METHOD USED: {saliency_method}
• HEATMAP: Red/Yellow = High attention areas
• DeepGaze shows human-like attention patterns"""
axs[1, 1].text(0.5, 0.5, interpretation, ha='center', va='center',
               fontsize=12, wrap=True, bbox=dict(facecolor='whitesmoke', alpha=0.8))
axs[1, 1].axis("off")

plt.tight_layout()
output_filename = f"ad_analysis_{TIMESTAMP}.png"
plt.savefig(output_filename, dpi=200, bbox_inches='tight')
plt.show()

print(f"Analysis saved to: {output_filename}")








# import cv2
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from PIL import Image
# import easyocr
# from torchvision import transforms
# import deepgaze_pytorch
# from datetime import datetime
# import warnings
#
# # CONFIG
# IMAGE_PATH = "Audi_CR08545138718757879809_v1_image.png"
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# CTA_KEYWORDS = ["learn more", "book a test drive", "shop now", "order", "sign up", "try now", "explore", "test drive"]
# TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
#
# # Load image
# image_bgr = cv2.imread(IMAGE_PATH)
# if image_bgr is None:
#     raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# resized_rgb = cv2.resize(image_rgb, (1024, 768))  # W x H
#
#
# # ---------- OpenCV Saliency (Robust Implementation) ----------
# def compute_opencv_saliency(image):
#     """Handle all possible OpenCV saliency method scenarios"""
#     saliency_map = None
#     method_used = "None"
#
#     try:
#         # Try Spectral Residual (most common)
#         if hasattr(cv2, 'saliency') and 'StaticSaliencySpectralResidual_create' in dir(cv2.saliency):
#             saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
#             success, saliency_map = saliency.computeSaliency(image)
#             if success:
#                 method_used = "SpectralResidual"
#                 return saliency_map, method_used
#
#         # Try Fine Grained (older versions)
#         if hasattr(cv2, 'saliency') and 'StaticSaliencyFineGrained_create' in dir(cv2.saliency):
#             saliency = cv2.saliency.StaticSaliencyFineGrained_create()
#             success, saliency_map = saliency.computeSaliency(image)
#             if success:
#                 method_used = "FineGrained"
#                 return saliency_map, method_used
#
#         # Try Objectness as last resort
#         if hasattr(cv2, 'saliency') and 'ObjectnessBING_create' in dir(cv2.saliency):
#             saliency = cv2.saliency.ObjectnessBING_create()
#             success, saliency_map = saliency.computeSaliency(image)
#             if success:
#                 method_used = "Objectness"
#                 return saliency_map, method_used
#
#     except Exception as e:
#         warnings.warn(f"Saliency computation failed: {str(e)}")
#
#     # Fallback: Create center-weighted dummy map
#     h, w = image.shape[:2]
#     saliency_map = np.zeros((h, w), dtype=np.float32)
#     cx, cy = w // 2, h // 2
#     cv2.circle(saliency_map, (cx, cy), min(w, h) // 3, 1.0, -1)
#     method_used = "DummyFallback"
#
#     return saliency_map, method_used
#
#
# saliency_map, saliency_method = compute_opencv_saliency(resized_rgb)
# print(f"Used saliency method: {saliency_method}")
# saliency_map_opencv = (saliency_map * 255).astype("uint8")
#
# # ---------- DeepGaze III ----------
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
# # ---------- EasyOCR CTA Detection (Updated) ----------
# # Initialize EasyOCR with explicit settings
# reader = easyocr.Reader(
#     ['en'],
#     gpu=torch.cuda.is_available(),
#     model_storage_directory=None,
#     download_enabled=True
# )
#
#
# def detect_cta_easyocr(image_rgb):
#     # Convert to BGR for EasyOCR
#     image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
#     results = reader.readtext(image_bgr)
#     boxes = []
#     for (bbox, text, conf) in results:
#         word = text.strip().lower()
#         if any(keyword in word for keyword in CTA_KEYWORDS):
#             x1, y1 = map(int, bbox[0])
#             x2, y2 = map(int, bbox[2])
#             boxes.append((x1, y1, x2, y2, word, conf))
#     return boxes
#
#
# cta_boxes = detect_cta_easyocr(resized_rgb)
#
#
# # ---------- Visualization Functions ----------
# def draw_cta_boxes(image, boxes, color=(0, 255, 0)):
#     img_copy = image.copy()
#     for (x1, y1, x2, y2, word, conf) in boxes:
#         cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
#         label = f"{word} ({conf:.2f})"
#         cv2.putText(img_copy, label, (x1, max(0, y1 - 5)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#     return img_copy
#
#
# def highlight_attention(image, saliency_map, alpha=0.6):
#     """Highlight attention zones with red/yellow colormap"""
#     norm_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
#     heatmap = cv2.applyColorMap((norm_map * 255).astype('uint8'), cv2.COLORMAP_HOT)
#     heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     blended = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)
#     return blended
#
#
# # Create visualizations
# image_with_cta = draw_cta_boxes(resized_rgb, cta_boxes)
# opencv_attention = highlight_attention(resized_rgb, saliency_map)
# deepgaze_attention = highlight_attention(resized_rgb, deepgaze_map)
#
# # Add CTA boxes to attention maps
# for box in cta_boxes:
#     x1, y1, x2, y2, word, conf = box
#     cv2.rectangle(opencv_attention, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.rectangle(deepgaze_attention, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# # ---------- Enhanced Visualization ----------
# fig, axs = plt.subplots(2, 2, figsize=(20, 12))
#
# # Original with CTAs
# axs[0, 0].imshow(image_with_cta)
# axs[0, 0].set_title("1. CTA Detection (EasyOCR)")
# axs[0, 0].axis("off")
#
# # OpenCV Saliency
# axs[0, 1].imshow(opencv_attention)
# axs[0, 1].set_title(f"2. OpenCV Saliency ({saliency_method})")
# axs[0, 1].axis("off")
#
# # DeepGaze III
# axs[1, 0].imshow(deepgaze_attention)
# axs[1, 0].set_title("3. DeepGaze III with CTA")
# axs[1, 0].axis("off")
#
# # Interpretation
# interpretation = f"""INTERPRETATION GUIDE:
# • GREEN BOXES: Detected Call-to-Action (CTA) elements
# • SALIENCY METHOD USED: {saliency_method}
# • HEATMAP: Red/Yellow = High attention areas
# • DeepGaze shows human-like attention patterns"""
# axs[1, 1].text(0.5, 0.5, interpretation, ha='center', va='center',
#                fontsize=12, wrap=True, bbox=dict(facecolor='whitesmoke', alpha=0.8))
# axs[1, 1].axis("off")
#
# plt.tight_layout()
# output_filename = f"ad_analysis_{TIMESTAMP}.png"
# plt.savefig(output_filename, dpi=200, bbox_inches='tight')
# plt.show()
#
# print(f"Analysis saved to: {output_filename}")