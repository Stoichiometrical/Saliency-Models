#Generates PDF, giving poor results
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import deepgaze_pytorch
from scipy.stats import entropy
from fpdf import FPDF
import tempfile

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

    width, height = 1024, 768
    fixation_history_x = np.array([width // 2, width // 4, 3 * width // 4, width // 3], dtype=np.float32)
    fixation_history_y = np.array([height // 2, height // 4, 3 * height // 4, height // 3], dtype=np.float32)

    x_hist = torch.from_numpy(fixation_history_x[np.newaxis, :]).float().to(device)
    y_hist = torch.from_numpy(fixation_history_y[np.newaxis, :]).float().to(device)

    with torch.no_grad():
        output = model(image_tensor, centerbias, x_hist=x_hist, y_hist=y_hist)

    saliency_map = output.squeeze().cpu().numpy()
    return image, saliency_map, fixation_history_x, fixation_history_y

# Function to calculate clarity and entropy
def compute_metrics(saliency_map):
    normalized_map = saliency_map / np.max(saliency_map)
    clarity = np.max(normalized_map)
    saliency_flat = normalized_map.flatten() + 1e-8
    ent = entropy(saliency_flat, base=2)
    return clarity, ent

# Ad paths
ad_folder = "ads"
ad_files = [
    "Audi_CR08545138718757879809_v1_image.png",
    "Audi_CR08545138718757879809_v2_image.png",
    "BMW_CR09799707874329362433_v2_image.png",
    "Audi_CR08545138718757879809_v1_image.png",
]

# Process each ad
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

# Create PDF report
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Ad Attention Analysis Report", ln=True, align='C')

# Summary table
pdf.set_font("Arial", '', 12)
pdf.ln(10)
pdf.cell(80, 10, "Ad Name", border=1)
pdf.cell(40, 10, "Clarity", border=1)
pdf.cell(40, 10, "Entropy", border=1)
pdf.ln()

for ad_name, data in results.items():
    clarity = data['clarity']
    entropy_score = data['entropy']
    pdf.cell(80, 10, ad_name[:40], border=1)  # Limit long names
    pdf.cell(40, 10, f"{clarity:.3f}", border=1)
    pdf.cell(40, 10, f"{entropy_score:.2f}", border=1)
    pdf.ln()

# Visuals
for ad_name, data in results.items():
    image = data['image']
    saliency_map = data['saliency_map']
    fix_x = data['fix_x']
    fix_y = data['fix_y']

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(image)
    axs[0].plot(fix_x, fix_y, 'ro-')
    axs[0].axis('off')
    axs[0].set_title("Scanpath")

    axs[1].imshow(image)
    axs[1].imshow(saliency_map, cmap='hot', alpha=0.5)
    axs[1].axis('off')
    axs[1].set_title("Saliency Overlay")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        fig.savefig(tmpfile.name, bbox_inches='tight')
        plt.close(fig)
        pdf.add_page()
        pdf.image(tmpfile.name, x=10, y=20, w=190)

# Save the final PDF
output_pdf_path = "Ad_Attention_Analysis_Report.pdf"
pdf.output(output_pdf_path)

output_pdf_path
