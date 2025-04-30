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


def plot_red_blue_saliency(image, saliency_map, ax, title):
    """Visualize with red (hot) and blue (cold) zones - CORRECTED VERSION"""
    # Thresholds (now correctly mapped)
    hot_thresh = np.percentile(saliency_map, 90)  # Top 10% (hot - red)
    cold_thresh = np.percentile(saliency_map, 10)  # Bottom 10% (cold - blue)

    # Create inverted colormaps
    hot_cmap = LinearSegmentedColormap.from_list('hot_cmap', ['black', 'red'])
    cold_cmap = LinearSegmentedColormap.from_list('cold_cmap', ['blue', 'black'])

    # Show original image
    ax.imshow(image, alpha=0.7)  # Slightly more transparent

    # Plot HOT zones (red) - now correctly mapped to high values
    hot_mask = saliency_map >= hot_thresh
    hot_plot = ax.imshow(
        np.ma.masked_where(~hot_mask, saliency_map),
        cmap=hot_cmap,
        vmin=hot_thresh,
        vmax=saliency_map.max(),
        alpha=0.8  # More opaque for emphasis
    )

    # Plot COLD zones (blue) - now correctly mapped to low values
    cold_mask = saliency_map <= cold_thresh
    cold_plot = ax.imshow(
        np.ma.masked_where(~cold_mask, saliency_map),
        cmap=cold_cmap,
        vmin=saliency_map.min(),
        vmax=cold_thresh,
        alpha=0.5  # Less opaque
    )

    # Add colorbars with corrected labels
    plt.colorbar(hot_plot, ax=ax, fraction=0.04, pad=0.01,
                 label='High Attention (Top 10%)')
    plt.colorbar(cold_plot, ax=ax, fraction=0.04, pad=0.01,
                 label='Low Attention (Bottom 10%)')

    ax.set_title(title, pad=20, fontsize=12, weight='bold')
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

# ===== Visualization =====
fig, axs = plt.subplots(len(results), 2, figsize=(20, 8 * len(results)))

for idx, (ad_name, data) in enumerate(results.items()):
    # Original image with scanpath
    axs[idx, 0].imshow(data['image'])
    axs[idx, 0].plot(data['fix_x'], data['fix_y'], 'yo-', markersize=8, linewidth=2)
    axs[idx, 0].scatter(data['fix_x'][-1], data['fix_y'][-1], 100, color='lime', zorder=100)
    axs[idx, 0].set_title(
        f"{ad_name}\nClarity: {data['clarity']:.2f} | Entropy: {data['entropy']:.2f}",
        pad=15,
        fontsize=12
    )
    axs[idx, 0].axis('off')

    # Red/Blue saliency map
    plot_red_blue_saliency(
        data['image'],
        data['saliency_map'],
        axs[idx, 1],
        f"Attention Map\nRed=Hot Zones | Blue=Cold Zones"
    )

    # Mark attention zone
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