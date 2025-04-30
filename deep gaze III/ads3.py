# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import deepgaze_pytorch
# from scipy.stats import entropy
# from scipy.ndimage import gaussian_filter
# from matplotlib.patches import Rectangle
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
#     saliency_map = gaussian_filter(saliency_map, sigma=5)
#     saliency_map = saliency_map / np.max(saliency_map)  # Normalize
#
#     return image, saliency_map, fixation_history_x, fixation_history_y
#
# # Function to calculate clarity and entropy
# def compute_metrics(saliency_map):
#     clarity = np.max(saliency_map)
#     saliency_flat = saliency_map.flatten() + 1e-8  # Prevent log(0)
#     ent = entropy(saliency_flat, base=2)
#     return clarity, ent
#
# # === Your Ads ===
# ad_folder = "ads"
# ad_files = [
#     "Audi_CR08545138718757879809_v1_image.png",
#     "Audi_CR08545138718757879809_v2_image.png",
#     "BMW_CR09799707874329362433_v2_image.png",
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
# # === Visualization with overlays and warnings ===
# fig, axs = plt.subplots(len(results), 2, figsize=(16, 5 * len(results)))
#
# for idx, (ad_name, data) in enumerate(results.items()):
#     image = data['image']
#     saliency_map = data['saliency_map']
#     fix_x = data['fix_x']
#     fix_y = data['fix_y']
#     clarity = data['clarity']
#     entropy_score = data['entropy']
#
#     # First column: original image with fixations
#     axs[idx, 0].imshow(image)
#     axs[idx, 0].plot(fix_x, fix_y, 'ro-')
#     axs[idx, 0].set_title(f"{ad_name}\nClarity={clarity:.3f} | Entropy={entropy_score:.2f}")
#     axs[idx, 0].axis('off')
#
#     # === Overlay visualization ===
#     axs[idx, 1].imshow(image)
#     axs[idx, 1].imshow(saliency_map, cmap='jet', alpha=0.6)
#
#     # --- Draw high-attention zone ---
#     threshold = np.percentile(saliency_map, 90)
#     mask = saliency_map >= threshold
#     rows, cols = np.where(mask)
#     if rows.size > 0 and cols.size > 0:
#         top, bottom = rows.min(), rows.max()
#         left, right = cols.min(), cols.max()
#         axs[idx, 1].add_patch(Rectangle((left, top), right - left, bottom - top,
#                                         edgecolor='lime', linewidth=2, fill=False, label='Attention Zone'))
#
#         # --- Define CTA zone (bottom 15%) ---
#         cta_top = int(0.85 * saliency_map.shape[0])
#         cta_bottom = saliency_map.shape[0]
#         cta_left = 0
#         cta_right = saliency_map.shape[1]
#
#         axs[idx, 1].add_patch(Rectangle((cta_left, cta_top), cta_right - cta_left, cta_bottom - cta_top,
#                                         edgecolor='red', linewidth=2, fill=False, linestyle='--', label='CTA Zone'))
#
#         # --- Check for mismatch ---
#         if bottom < cta_top:
#             axs[idx, 1].text(10, 30, "⚠️ CTA Not in Attention Zone", color='red', fontsize=12, backgroundcolor='white')
#
#     axs[idx, 1].set_title(f"Saliency Overlay - {ad_name}")
#     axs[idx, 1].axis('off')
#
# plt.tight_layout()
# plt.show()


#WORKS WITH YELLOW AND RED SALIENCE ZONES
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import deepgaze_pytorch
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

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


def process_ad(image_path):
    """Process image and generate saliency map"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    centerbias = torch.zeros((1, 768, 1024)).to(device)

    # Synthetic fixation history
    width, height = 1024, 768
    fixation_history_x = np.array([
        width // 2, width // 4, 3 * width // 4, width // 3
    ], dtype=np.float32)
    fixation_history_y = np.array([
        height // 2, height // 4, 3 * height // 4, height // 3
    ], dtype=np.float32)

    x_hist = torch.from_numpy(fixation_history_x[np.newaxis, :]).float().to(device)
    y_hist = torch.from_numpy(fixation_history_y[np.newaxis, :]).float().to(device)

    # Generate saliency map
    with torch.no_grad():
        output = model(image_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)

    saliency_map = output.squeeze().cpu().numpy()
    saliency_map = gaussian_filter(saliency_map, sigma=5)
    saliency_map = saliency_map / np.max(saliency_map)  # Normalize

    return image, saliency_map, fixation_history_x, fixation_history_y


def compute_metrics(saliency_map):
    """Calculate clarity and entropy metrics"""
    clarity = np.max(saliency_map)
    saliency_flat = saliency_map.flatten() + 1e-8  # Prevent log(0)
    ent = entropy(saliency_flat, base=2)
    return clarity, ent


def plot_saliency_contrast(image, saliency_map, ax, title):
    """Visualize only high/low saliency areas with custom colormaps"""
    # Thresholds for high/low saliency
    high_thresh = np.percentile(saliency_map, 90)  # Top 10%
    low_thresh = np.percentile(saliency_map, 10)  # Bottom 10%

    # Create masks
    high_mask = saliency_map >= high_thresh
    low_mask = saliency_map <= low_thresh

    # Show original image
    ax.imshow(image)

    # Plot high saliency in red
    high_saliency = np.ma.masked_where(~high_mask, saliency_map)
    im1 = ax.imshow(high_saliency, cmap='Reds', vmin=high_thresh, alpha=0.7)

    # Plot low saliency in yellow
    low_saliency = np.ma.masked_where(~low_mask, saliency_map)
    im2 = ax.imshow(low_saliency, cmap='YlOrBr', vmax=low_thresh, alpha=0.5)

    # Add colorbars
    plt.colorbar(im1, ax=ax, fraction=0.04, pad=0.01, label='High Saliency')
    plt.colorbar(im2, ax=ax, fraction=0.04, pad=0.01, label='Low Saliency')

    ax.set_title(title, pad=20)
    ax.axis('off')


# ===== Main Analysis =====
ad_folder = "ads"
ad_files = [
    "Audi_CR08545138718757879809_v1_image.png",
    "Audi_CR08545138718757879809_v2_image.png",
    "BMW_CR09799707874329362433_v2_image.png",
]

results = {}
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

# ===== Enhanced Visualization =====
fig, axs = plt.subplots(len(results), 2, figsize=(20, 8 * len(results)))

for idx, (ad_name, data) in enumerate(results.items()):
    # Original image with scanpath
    axs[idx, 0].imshow(data['image'])
    axs[idx, 0].plot(data['fix_x'], data['fix_y'], 'ro-', markersize=8, linewidth=2)
    axs[idx, 0].scatter(data['fix_x'][-1], data['fix_y'][-1], 100, color='lime', zorder=100)
    axs[idx, 0].set_title(f"{ad_name}\nClarity: {data['clarity']:.2f} | Entropy: {data['entropy']:.2f}", pad=15)
    axs[idx, 0].axis('off')

    # Enhanced saliency contrast
    plot_saliency_contrast(
        data['image'],
        data['saliency_map'],
        axs[idx, 1],
        f"Saliency Contrast Map\n(Red=Top 10%, Yellow=Bottom 10%)"
    )

    # Add attention zone box
    threshold = np.percentile(data['saliency_map'], 90)
    mask = data['saliency_map'] >= threshold
    if mask.any():
        rows, cols = np.where(mask)
        top, bottom = rows.min(), rows.max()
        left, right = cols.min(), cols.max()
        axs[idx, 1].add_patch(Rectangle(
            (left, top), right - left, bottom - top,
            edgecolor='lime', linewidth=3, fill=False, linestyle='--'
        ))

plt.tight_layout()
plt.show()
