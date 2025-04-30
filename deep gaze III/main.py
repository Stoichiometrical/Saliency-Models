# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import sys
#
# # Assuming you have DeepGazeIII already cloned
# # sys.path.append('/path/to/DeepGaze')  # update this path
#
# import deepgaze_pytorch
#
# # Device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# # Load model
# model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
# model.eval()
#
# # Load ad image
# image_path = "Audi_CR08545138718757879809_v1_image.png"
# image = Image.open(image_path).convert('RGB')
#
# # Preprocess
# transform = transforms.Compose([
#     transforms.Resize((768, 1024)),
#     transforms.ToTensor(),
# ])
# image_tensor = transform(image).unsqueeze(0).to(device)
#
# centerbias = torch.zeros((1, 1, 768, 1024)).to(device)
# # Predict saliency
# with torch.no_grad():
#     output = model(image_tensor, centerbias)
#
#
# saliency_map = output.squeeze().cpu().numpy()
#
# # Display heatmap
# plt.imshow(saliency_map, cmap='hot')
# plt.title("DeepGaze III Attention Heatmap")
# plt.axis('off')
# plt.show()
import math


#Work but with convoluted saliency map
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import deepgaze_pytorch

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
model.eval()

# Image preprocessing - must match model expectations
image_path = "Audi_CR08545138718757879809_v1_image.png"
image = Image.open(image_path).convert('RGB')

# Note: DeepGazeIII expects (height, width) = (768, 1024)
transform = transforms.Compose([
    transforms.Resize((768, 1024)),  # Height, Width
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0).to(device)  # Shape: [1, 3, 768, 1024]

# Create proper centerbias tensor - must be [1, height, width]
centerbias = torch.zeros((1, 768, 1024)).to(device)

# Create fixation history - must provide exactly 4 fixations
height, width = 768, 1024
fixation_history_x = np.array([
    width // 2,      # Center
    width // 4,      # Left
    3 * width // 4,  # Right
    width // 3       # Middle-left
], dtype=np.float32)

fixation_history_y = np.array([
    height // 2,      # Center
    height // 4,      # Top
    3 * height // 4,  # Bottom
    height // 3       # Middle-top
], dtype=np.float32)

# Convert to tensors with correct shape [1, 4]
x_hist = torch.from_numpy(fixation_history_x[np.newaxis, :]).float().to(device)
y_hist = torch.from_numpy(fixation_history_y[np.newaxis, :]).float().to(device)

# Prediction
with torch.no_grad():
    output = model(
        image_tensor,
        centerbias,
        x_hist=x_hist,
        y_hist=y_hist
    )

# Process output
saliency_map = output.squeeze().cpu().numpy()  # Shape: [768, 1024]

# Visualization
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.plot(fixation_history_x, fixation_history_y, 'ro-')
plt.title("Original Image with Scanpath")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap='hot')
plt.title("DeepGazeIII Saliency Prediction")
plt.axis('off')
plt.tight_layout()
plt.show()