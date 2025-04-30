# import os
# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import torchvision.transforms as transforms
# import deepgaze_pytorch
# from scipy.ndimage import gaussian_filter
# from matplotlib.colors import LinearSegmentedColormap
#
# # Initialize models with error handling
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# try:
#     deepgaze_model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
#     deepgaze_model.eval()
# except Exception as e:
#     raise RuntimeError(f"Failed to load DeepGaze model: {str(e)}")
#
# try:
#     opencv_saliency = cv2.saliency.StaticSaliencyFineGrained_create()
#     if opencv_saliency is None:
#         raise RuntimeError("OpenCV saliency model not available (requires contrib modules)")
# except Exception as e:
#     raise RuntimeError(f"OpenCV saliency initialization failed: {str(e)}")
#
#
# def analyze_ad_hybrid(image_path):
#     """
#     Process an advertisement image using both OpenCV and DeepGaze saliency detection,
#     then combine results to highlight only extreme attention areas.
#
#     Args:
#         image_path: Path to the input image file
#
#     Returns:
#         Dictionary containing:
#         - original: Original RGB image
#         - opencv_heatmap: OpenCV-generated saliency map
#         - deepgaze_heatmap: DeepGaze-generated saliency map
#         - combined_heatmap: Fusion of both methods
#         - metrics: Dictionary of calculated metrics
#         - extremes_map: Visualization showing only high/low saliency areas
#     """
#     # ===== 1. Image Loading and Preprocessing =====
#     try:
#         # Load image via OpenCV (for OpenCV processing)
#         image_cv = cv2.imread(image_path)
#         if image_cv is None:
#             raise FileNotFoundError(f"Could not load image at {image_path}")
#         image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
#
#         # Load image via PIL (for DeepGaze processing)
#         image_pil = Image.open(image_path).convert('RGB')
#     except Exception as e:
#         raise RuntimeError(f"Image loading failed: {str(e)}")
#
#     # ===== 2. OpenCV Saliency Analysis =====
#     try:
#         # Generate base saliency map
#         success, opencv_heatmap = opencv_saliency.computeSaliency(image_cv)
#         if not success:
#             raise RuntimeError("OpenCV saliency computation failed")
#
#         # Convert and smooth heatmap
#         opencv_heatmap = (opencv_heatmap * 255).astype("uint8")
#         opencv_heatmap = cv2.GaussianBlur(opencv_heatmap, (25, 25), 0)
#     except Exception as e:
#         raise RuntimeError(f"OpenCV saliency processing failed: {str(e)}")
#
#     # ===== 3. DeepGaze Saliency Analysis =====
#     try:
#         # Prepare image tensor
#         transform = transforms.Compose([
#             transforms.Resize((768, 1024)),  # DeepGaze expected input size
#             transforms.ToTensor(),
#         ])
#         image_tensor = transform(image_pil).unsqueeze(0).to(device)
#         centerbias = torch.zeros((1, 768, 1024)).to(device)
#
#         # Synthetic fixation points (center and quadrants)
#         width, height = 1024, 768
#         x_hist = torch.tensor([[width // 2, width // 4, 3 * width // 4, width // 3]]).float().to(device)
#         y_hist = torch.tensor([[height // 2, height // 4, 3 * height // 4, height // 3]]).float().to(device)
#
#         # Generate prediction
#         with torch.no_grad():
#             deepgaze_output = deepgaze_model(image_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)
#
#         # Post-process heatmap
#         deepgaze_heatmap = deepgaze_output.squeeze().cpu().numpy()
#         deepgaze_heatmap = gaussian_filter(deepgaze_heatmap, sigma=5)
#         deepgaze_heatmap = (deepgaze_heatmap - deepgaze_heatmap.min()) / (
#                     deepgaze_heatmap.max() - deepgaze_heatmap.min())  # Normalize 0-1
#     except Exception as e:
#         raise RuntimeError(f"DeepGaze processing failed: {str(e)}")
#
#     # ===== 4. Data Fusion =====
#     try:
#         # Resize OpenCV heatmap to match DeepGaze and normalize
#         opencv_resized = cv2.resize(opencv_heatmap, (1024, 768)) / 255.0
#
#         # Combine heatmaps (weighted average favoring DeepGaze)
#         combined_heatmap = 0.7 * deepgaze_heatmap + 0.3 * opencv_resized
#         combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (
#                     combined_heatmap.max() - combined_heatmap.min())  # Renormalize
#
#         # Create visualization showing only extreme values
#         hot_thresh = np.percentile(combined_heatmap, 90)  # Top 10%
#         cold_thresh = np.percentile(combined_heatmap, 10)  # Bottom 10%
#
#         # Create masks
#         hot_mask = combined_heatmap >= hot_thresh
#         cold_mask = combined_heatmap <= cold_thresh
#
#         # Create colormaps
#         hot_cmap = LinearSegmentedColormap.from_list('hot_cmap', ['transparent', 'red'])
#         cold_cmap = LinearSegmentedColormap.from_list('cold_cmap', ['blue', 'transparent'])
#
#         # Prepare visualization
#         extremes_map = image_rgb.copy()
#         plt.figure(figsize=(10, 10))
#         plt.imshow(extremes_map)
#
#         # Overlay hot zones (red)
#         plt.imshow(
#             np.ma.masked_where(~hot_mask, combined_heatmap),
#             cmap=hot_cmap,
#             vmin=hot_thresh,
#             alpha=0.6  # Semi-transparent
#         )
#
#         # Overlay cold zones (blue)
#         plt.imshow(
#             np.ma.masked_where(~cold_mask, combined_heatmap),
#             cmap=cold_cmap,
#             vmax=cold_thresh,
#             alpha=0.4  # More transparent
#         )
#         plt.axis('off')
#         plt.close()  # Prevent automatic display
#
#     except Exception as e:
#         raise RuntimeError(f"Data fusion failed: {str(e)}")
#
#     # ===== 5. Metric Calculation =====
#     metrics = {
#         "Clarity": np.max(combined_heatmap),  # Peak attention strength (0-1)
#         "Focus": np.std(combined_heatmap),  # Concentration of attention
#         "Engagement": np.mean(combined_heatmap),  # Overall attention level
#         "Hot Coverage": np.mean(hot_mask) * 100,  # % of high-attention area
#         "Cold Coverage": np.mean(cold_mask) * 100  # % of low-attention area
#     }
#
#     return {
#         "original": image_rgb,
#         "opencv_heatmap": opencv_heatmap,
#         "deepgaze_heatmap": (deepgaze_heatmap * 255).astype(np.uint8),
#         "combined_heatmap": (combined_heatmap * 255).astype(np.uint8),
#         "extremes_map": plt.gcf(),  # Contains the visualization figure
#         "metrics": metrics
#     }
#
#
# def plot_results(results):
#     """
#     Visualize comparison between original ad and saliency analysis.
#
#     Args:
#         results: Dictionary containing analysis results from analyze_ad_hybrid()
#     """
#     plt.figure(figsize=(18, 6 * len(results)))
#
#     for idx, (name, data) in enumerate(results.items()):
#         # Row positions
#         row_start = idx * 2
#
#         # Column 1: Original Image
#         plt.subplot(len(results), 3, row_start * 3 + 1)
#         plt.imshow(data['original'])
#         plt.title(f"{name}\nOriginal Ad", fontsize=12)
#         plt.axis('off')
#
#         # Column 2: Extreme Saliency Visualization
#         plt.subplot(len(results), 3, row_start * 3 + 2)
#         # Recreate the extremes map visualization
#         plt.imshow(data['original'], alpha=0.8)
#
#         # Get thresholds from metrics
#         hot_thresh = np.percentile(data['combined_heatmap'] / 255, 90)
#         cold_thresh = np.percentile(data['combined_heatmap'] / 255, 10)
#
#         # Plot hot zones
#         hot_mask = data['combined_heatmap'] / 255 >= hot_thresh
#         plt.imshow(
#             np.ma.masked_where(~hot_mask, data['combined_heatmap'] / 255),
#             cmap='Reds',
#             vmin=hot_thresh,
#             alpha=0.6
#         )
#
#         # Plot cold zones
#         cold_mask = data['combined_heatmap'] / 255 <= cold_thresh
#         plt.imshow(
#             np.ma.masked_where(~cold_mask, data['combined_heatmap'] / 255),
#             cmap='Blues_r',  # Reversed to make blue = cold
#             vmax=cold_thresh,
#             alpha=0.4
#         )
#
#         # Add metrics
#         metrics_text = "\n".join([
#             f"Hot Area: {data['metrics']['Hot Coverage']:.1f}%",
#             f"Cold Area: {data['metrics']['Cold Coverage']:.1f}%",
#             f"Engagement: {data['metrics']['Engagement']:.2f}"
#         ])
#         plt.text(
#             20, 40, metrics_text,
#             color='white', fontsize=10,
#             bbox=dict(facecolor='black', alpha=0.6))
#
#         plt.title("Attention Extremes\n(Red=Top 10%, Blue=Bottom 10%)", fontsize=12)
#         plt.axis('off')
#
#         # Column 3: Metrics Table
#         plt.subplot(len(results), 3, row_start * 3 + 3)
#         plt.axis('off')
#         plt.table(
#             cellText=[[f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
#                        for k, v in data['metrics'].items()]],
#             colLabels=["Metric"],
#             loc="center",
#             cellLoc="left"
#         )
#         plt.title("Performance Metrics", fontsize=12)
#
#         plt.tight_layout()
#         plt.show()
#
#         # ===== Main Execution =====
# if __name__ == "__main__":
#             # Define ad paths (replace with your actual paths)
#             ad_paths = {
#                 "Our_Ad": "ads/Audi_CR08545138718757879809_v1_image.png",
#                 "Competitor_Ad": "ads/BMW_CR09799707874329362433_v2_image.png"
#             }
#
#             try:
#                 # Process all ads
#                 results = {}
#                 for name, path in ad_paths.items():
#                     print(f"Processing {name}...")
#                     results[name] = analyze_ad_hybrid(path)
#
#                 # Visualize results
#                 plot_results(results)
#
#             except Exception as e:
#                 print(f"Error during analysis: {str(e)}")


import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import deepgaze_pytorch
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

"""
AD SALIENCY ANALYSIS SCRIPT

This script analyzes advertisement images using a hybrid approach combining:
1. DeepGazeIII (neural network-based saliency prediction)
2. OpenCV's fine-grained saliency detection

Key outputs:
- Full attention heatmap overlay
- Top/bottom attention zones (10% thresholds)
- Quantitative metrics about visual engagement

Interpretation guide:
- Red areas: Highest predicted attention zones (top 10%)
- Blue areas: Lowest predicted attention zones (bottom 10%)
- Yellow/white in full heatmap: Areas likely to attract attention
- Dark blue/purple: Areas likely to be overlooked
- Higher metrics generally indicate better visual engagement
"""

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
deepgaze_model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
deepgaze_model.eval()
opencv_saliency = cv2.saliency.StaticSaliencyFineGrained_create()


def analyze_ad_hybrid(image_path):
    """Process an ad image using OpenCV and DeepGaze saliency detection

    Args:
        image_path: Path to the advertisement image file

    Returns:
        Dictionary containing:
        - original: Original RGB image
        - combined_heatmap: Normalized hybrid saliency map
        - metrics: Dictionary of quantitative metrics
        - original_size: Original image dimensions for proper resizing
    """
    try:
        # Load and prepare images
        # OpenCV loads in BGR format by default
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Store original dimensions for proper resizing later
        original_height, original_width = image_cv.shape[:2]

        # Convert to RGB for display purposes
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.open(image_path).convert('RGB')

        # OpenCV saliency computation
        # This provides a basic low-level saliency map
        success, opencv_heatmap = opencv_saliency.computeSaliency(image_cv)
        if not success:
            raise RuntimeError("OpenCV saliency computation failed")

        # Normalize and smooth the OpenCV heatmap
        opencv_heatmap = (opencv_heatmap * 255).astype("uint8")
        opencv_heatmap = cv2.GaussianBlur(opencv_heatmap, (25, 25), 0)

        # DeepGaze processing - requires specific input size (1024x768)
        # We'll transform the image to this size but keep track of original dimensions
        transform = transforms.Compose([
            transforms.Resize((768, 1024)),  # DeepGaze expects this exact size
            transforms.ToTensor(),
        ])
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        # Center bias - models human tendency to look at center
        centerbias = torch.zeros((1, 768, 1024)).to(device)

        # Historical gaze positions (simulated)
        width, height = 1024, 768
        x_hist = torch.tensor([[width // 2, width // 4, 3 * width // 4, width // 3]]).float().to(device)
        y_hist = torch.tensor([[height // 2, height // 4, 3 * height // 4, height // 3]]).float().to(device)

        # Run DeepGaze prediction
        with torch.no_grad():
            deepgaze_output = deepgaze_model(image_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)

        # Process DeepGaze output
        deepgaze_heatmap = deepgaze_output.squeeze().cpu().numpy()
        deepgaze_heatmap = gaussian_filter(deepgaze_heatmap, sigma=5)  # Smooth the heatmap
        deepgaze_heatmap = (deepgaze_heatmap - deepgaze_heatmap.min()) / (
                deepgaze_heatmap.max() - deepgaze_heatmap.min())  # Normalize to 0-1

        # Combine heatmaps (70% DeepGaze, 30% OpenCV)
        opencv_resized = cv2.resize(opencv_heatmap, (1024, 768)) / 255.0
        combined_heatmap = 0.7 * deepgaze_heatmap + 0.3 * opencv_resized
        combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (
                combined_heatmap.max() - combined_heatmap.min())

        # Calculate metrics
        hot_thresh = np.percentile(combined_heatmap, 90)  # Top 10% threshold
        cold_thresh = np.percentile(combined_heatmap, 10)  # Bottom 10% threshold
        hot_mask = combined_heatmap >= hot_thresh
        cold_mask = combined_heatmap <= cold_thresh

        metrics = {
            "Clarity": np.max(combined_heatmap),  # Peak attention value
            "Focus": np.std(combined_heatmap),  # How concentrated attention is
            "Engagement": np.mean(combined_heatmap),  # Average attention
            "Hot Coverage": np.mean(hot_mask) * 100,  # % of top attention areas
            "Cold Coverage": np.mean(cold_mask) * 100  # % of bottom attention areas
        }

        return {
            "original": image_rgb,
            "combined_heatmap": combined_heatmap,
            "metrics": metrics,
            "original_size": (original_width, original_height),  # Store original dimensions
            "model_input_size": (1024, 768)  # Store model input dimensions
        }

    except Exception as e:
        raise RuntimeError(f"Error processing {image_path}: {str(e)}")


def plot_results(results):
    """Visualize analysis results with full attention heatmap

    Args:
        results: Dictionary of analysis results from analyze_ad_hybrid()
    """
    fig, axs = plt.subplots(len(results), 4, figsize=(24, 6 * len(results)))
    if len(results) == 1:
        axs = np.expand_dims(axs, axis=0)  # Ensure 2D array for single ad

    for idx, (name, data) in enumerate(results.items()):
        # Get original and model input dimensions
        orig_width, orig_height = data['original_size']
        model_width, model_height = data['model_input_size']

        # Resize heatmap back to original dimensions for proper overlay
        resized_heatmap = cv2.resize(data['combined_heatmap'],
                                     (orig_width, orig_height))

        # Original Image
        axs[idx, 0].imshow(data['original'])
        axs[idx, 0].set_title(f"{name}\nOriginal Ad", fontsize=12)
        axs[idx, 0].axis('off')

        # Full Attention Heatmap Overlay
        axs[idx, 1].imshow(data['original'], alpha=0.8)

        # Create custom colormap (red-yellow-blue)
        cmap = LinearSegmentedColormap.from_list('attention_cmap',
                                                 ['darkblue', 'blue', 'cyan',
                                                  'yellow', 'red'])

        # Show full heatmap with transparency
        heatmap_display = axs[idx, 1].imshow(
            resized_heatmap,
            cmap=cmap,
            alpha=0.6,
            vmin=0,
            vmax=1
        )

        # Add colorbar for reference
        plt.colorbar(heatmap_display, ax=axs[idx, 1], fraction=0.046, pad=0.04)

        metrics_text = "\n".join([
            f"Max Attention: {data['metrics']['Clarity']:.2f}",
            f"Engagement: {data['metrics']['Engagement']:.2f}",
            f"Focus: {data['metrics']['Focus']:.2f}"
        ])
        axs[idx, 1].text(
            20, 40, metrics_text,
            color='white', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.6))
        axs[idx, 1].set_title("Full Attention Heatmap\n(Yellow/Red=High, Blue=Low)", fontsize=12)
        axs[idx, 1].axis('off')

        # Attention Extremes (Top/Bottom 10%)
        axs[idx, 2].imshow(data['original'], alpha=0.8)

        # Calculate thresholds on resized heatmap
        hot_thresh = np.percentile(resized_heatmap, 90)
        cold_thresh = np.percentile(resized_heatmap, 10)

        # Hot zones (top 10%)
        hot_mask = resized_heatmap >= hot_thresh
        axs[idx, 2].imshow(
            np.ma.masked_where(~hot_mask, resized_heatmap),
            cmap='Reds',
            vmin=hot_thresh,
            alpha=0.7
        )

        # Cold zones (bottom 10%)
        cold_mask = resized_heatmap <= cold_thresh
        axs[idx, 2].imshow(
            np.ma.masked_where(~cold_mask, resized_heatmap),
            cmap='Blues_r',
            vmax=cold_thresh,
            alpha=0.5
        )

        extremes_text = "\n".join([
            f"Hot Area: {data['metrics']['Hot Coverage']:.1f}%",
            f"Cold Area: {data['metrics']['Cold Coverage']:.1f}%"
        ])
        axs[idx, 2].text(
            20, 40, extremes_text,
            color='white', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.6))
        axs[idx, 2].set_title("Attention Extremes\n(Red=Top 10%, Blue=Bottom 10%)", fontsize=12)
        axs[idx, 2].axis('off')

        # Metrics Table
        axs[idx, 3].axis('off')
        table_data = [[f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"]
                      for k, v in data['metrics'].items()]
        axs[idx, 3].table(
            cellText=table_data,
            colLabels=["Metric"],
            loc="center",
            cellLoc="left"
        )
        axs[idx, 3].set_title("Performance Metrics", fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage with sample ad paths
    ad_paths = {
        "Our_Ad": "ads/Audi_CR08545138718757879809_v1_image.png",
        "Competitor_Ad": "ads/BMW_CR09799707874329362433_v2_image.png"
    }

    try:
        results = {}
        for name, path in ad_paths.items():
            print(f"Processing {name}...")
            results[name] = analyze_ad_hybrid(path)

        plot_results(results)

    except Exception as e:
        print(f"Error during analysis: {str(e)}")