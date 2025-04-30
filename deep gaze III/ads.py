
#Higher clarity + lower entropy usually means better ad design for "grabbing" attention quickly!


# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import deepgaze_pytorch
# from scipy.stats import entropy
#
# # Device setup
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# # Load model
# model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
# model.eval()
#
# # Image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((768, 1024)),  # Height, Width
#     transforms.ToTensor(),
# ])
#
# # Function to process a single image
# def process_ad(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     centerbias = torch.zeros((1, 768, 1024)).to(device)
#
#     # Fixation history - synthetic
#     width, height = 1024, 768
#     fixation_history_x = np.array([
#         width // 2, width // 4, 3 * width // 4, width // 3
#     ], dtype=np.float32)
#     fixation_history_y = np.array([
#         height // 2, height // 4, 3 * height // 4, height // 3
#     ], dtype=np.float32)
#
#     x_hist = torch.from_numpy(fixation_history_x[np.newaxis, :]).float().to(device)
#     y_hist = torch.from_numpy(fixation_history_y[np.newaxis, :]).float().to(device)
#
#     # Prediction
#     with torch.no_grad():
#         output = model(image_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)
#
#     saliency_map = output.squeeze().cpu().numpy()
#
#     return image, saliency_map, fixation_history_x, fixation_history_y
#
# # Function to calculate clarity and entropy
# def compute_metrics(saliency_map):
#     normalized_map = saliency_map / np.max(saliency_map)
#     clarity = np.max(normalized_map)
#     saliency_flat = normalized_map.flatten() + 1e-8  # Prevent log(0)
#     ent = entropy(saliency_flat, base=2)
#     return clarity, ent
#
# # === Your Ads ===
# ad_folder = "ads"
# ad_files = [
#     "Audi_CR08545138718757879809_v1_image.png",
#     "Audi_CR08545138718757879809_v2_image.png",
#     "BMW_CR09799707874329362433_v2_image.png",
#     "Audi_CR08545138718757879809_v1_image.png",
# ]
#
# # Results
# results = {}
#
# # Loop through ads
# for ad_file in ad_files:
#     image_path = os.path.join(ad_folder, ad_file)
#     image, saliency_map, fix_x, fix_y = process_ad(image_path)
#     clarity, ent = compute_metrics(saliency_map)
#
#     results[ad_file] = {
#         "image": image,
#         "saliency_map": saliency_map,
#         "fix_x": fix_x,
#         "fix_y": fix_y,
#         "clarity": clarity,
#         "entropy": ent
#     }
#
# # === Visualization ===
# fig, axs = plt.subplots(len(results), 2, figsize=(16, 4 * len(results)))
#
# for idx, (ad_name, data) in enumerate(results.items()):
#     image = data['image']
#     saliency_map = data['saliency_map']
#     fix_x = data['fix_x']
#     fix_y = data['fix_y']
#     clarity = data['clarity']
#     entropy_score = data['entropy']
#
#     axs[idx, 0].imshow(image)
#     axs[idx, 0].plot(fix_x, fix_y, 'ro-')
#     axs[idx, 0].set_title(f"{ad_name}\nClarity={clarity:.3f} | Entropy={entropy_score:.2f}")
#     axs[idx, 0].axis('off')
#
#     axs[idx, 1].imshow(saliency_map, cmap='hot')
#     axs[idx, 1].set_title(f"Saliency Heatmap - {ad_name}")
#     axs[idx, 1].axis('off')
#
# plt.tight_layout()
# plt.show()


import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import deepgaze_pytorch
from scipy.stats import entropy

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((768, 1024)),  # Height, Width
    transforms.ToTensor(),
])

# Function to process a single image
def process_ad(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    centerbias = torch.zeros((1, 768, 1024)).to(device)

    # Fixation history - synthetic
    width, height = 1024, 768
    fixation_history_x = np.array([
        width // 2, width // 4, 3 * width // 4, width // 3
    ], dtype=np.float32)
    fixation_history_y = np.array([
        height // 2, height // 4, 3 * height // 4, height // 3
    ], dtype=np.float32)

    x_hist = torch.from_numpy(fixation_history_x[np.newaxis, :]).float().to(device)
    y_hist = torch.from_numpy(fixation_history_y[np.newaxis, :]).float().to(device)

    # Prediction
    with torch.no_grad():
        output = model(image_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)

    saliency_map = output.squeeze().cpu().numpy()

    return image, saliency_map, fixation_history_x, fixation_history_y

# Function to calculate clarity and entropy
def compute_metrics(saliency_map):
    normalized_map = saliency_map / np.max(saliency_map)
    clarity = np.max(normalized_map)
    saliency_flat = normalized_map.flatten() + 1e-8  # Prevent log(0)
    ent = entropy(saliency_flat, base=2)
    return clarity, ent

# === Your Ads ===
ad_folder = "ads"
ad_files = [
    "Audi_CR08545138718757879809_v1_image.png",
    "Audi_CR08545138718757879809_v2_image.png",
    "BMW_CR09799707874329362433_v2_image.png",
    "Audi_CR08545138718757879809_v1_image.png",
]

# Results
results = {}

# Loop through ads
for ad_file in ad_files:
    image_path = os.path.join(ad_folder, ad_file)
    image, saliency_map, fix_x, fix_y = process_ad(image_path)
    clarity, ent = compute_metrics(saliency_map)

    results[ad_file] = {
        "image": image,
        "saliency_map": saliency_map,
        "fix_x": fix_x,
        "fix_y": fix_y,
        "clarity": clarity,
        "entropy": ent
    }

# === Visualization (only this part is updated) ===
fig, axs = plt.subplots(len(results), 2, figsize=(16, 5 * len(results)))

for idx, (ad_name, data) in enumerate(results.items()):
    image = data['image']
    saliency_map = data['saliency_map']
    fix_x = data['fix_x']
    fix_y = data['fix_y']
    clarity = data['clarity']
    entropy_score = data['entropy']

    # First column: original image with fixation points
    axs[idx, 0].imshow(image)
    axs[idx, 0].plot(fix_x, fix_y, 'ro-')
    axs[idx, 0].set_title(f"{ad_name}\nClarity={clarity:.3f} | Entropy={entropy_score:.2f}")
    axs[idx, 0].axis('off')

    # Second column: original image + saliency heatmap overlay
    axs[idx, 1].imshow(image)  # Show original ad
    # axs[idx, 1].imshow(saliency_map, cmap='hot', alpha=0.5)  # Overlay heatmap with transparency
    axs[idx, 1].imshow(saliency_map, cmap='jet', alpha=0.6)
    axs[idx, 1].set_title(f"Saliency Overlay - {ad_name}")
    axs[idx, 1].axis('off')

plt.tight_layout()
plt.show()


