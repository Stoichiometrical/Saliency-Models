# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.ndimage import zoom
# from scipy.special import logsumexp
# import torch
# import sys
# from PIL import Image  # Add PIL to load the image
#
# # Add the directory containing deepgaze_pytorch to the Python path
# # sys.path.append('/content/DeepGaze')
#
# # Now import deepgaze_pytorch
# import deepgaze_pytorch
#
# DEVICE = 'cpu'
#
# # you can use DeepGazeI or DeepGazeIIE
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# # Load model
# model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
#
# # Load the image using PIL
# image_path = "BMW_CR09799707874329362433_v2_image.png"
# image = Image.open(image_path)
# image = np.array(image)  # Convert the image to a numpy array
# # If image has an alpha channel, convert it to RGB
# # if image.shape[2] == 4:
# #     image = image[:, :, :3]  # Keep only RGB channels
# # Check if the image is grayscale (2 dimensions) and convert to RGB if necessary
# if len(image.shape) == 2:  # Check if the image is grayscale
#     image = np.stack((image,) * 3, axis=-1)  # Convert grayscale to RGB by stacking 3 copies
# # If image has an alpha channel, convert it to RGB
# elif image.shape[2] == 4:
#     image = image[:, :, :3]  # Keep only RGB channels
#
#
#
# # location of previous scanpath fixations in x and y (pixel coordinates), starting with the initial fixation on the image.
# fixation_history_x = np.array([1024//2, 300, 500, 200, 200, 700])
# fixation_history_y = np.array([768//2, 300, 100, 300, 100, 500])
#
# # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
# # centerbias_template = np.load('centerbias_mit1003.npy')
# centerbias_template = np.zeros((1024, 1024))
# # rescale to match image size
# centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
# # renormalize log density
# centerbias -= logsumexp(centerbias)
#
# # Convert image to a tensor and move to the device
# image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
# centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
# x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
# y_hist_tensor = torch.tensor([fixation_history_y[model.included_fixations]]).to(DEVICE)
#
# # Get log density prediction
# log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
#
# # Plot the results
# f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
# axs[0].imshow(image)
# axs[0].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
# axs[0].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
# axs[0].set_axis_off()
# axs[1].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
# axs[1].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
# axs[1].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
# axs[1].set_axis_off()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import sys
from PIL import Image

# Import deepgaze_pytorch
import deepgaze_pytorch

DEVICE = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)

# Load the image
image_path = "Audi_CR08545138718757879809_v1_image.png"
image = Image.open(image_path)
image = np.array(image)

# Convert grayscale to RGB or remove alpha channel if needed
if len(image.shape) == 2:  # Grayscale
    image = np.stack((image,) * 3, axis=-1)
elif image.shape[2] == 4:  # RGBA
    image = image[:, :, :3]

# Fixation history
fixation_history_x = np.array([1024//2, 300, 500, 200, 200, 700])
fixation_history_y = np.array([768//2, 300, 100, 300, 100, 500])

# Create centerbias (placeholder)
centerbias_template = np.zeros((1024, 1024))
centerbias = zoom(centerbias_template,
                 (image.shape[0]/centerbias_template.shape[0],
                  image.shape[1]/centerbias_template.shape[1]),
                 order=0, mode='nearest')
centerbias -= logsumexp(centerbias)

# Convert to tensors
image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
y_hist_tensor = torch.tensor([fixation_history_y[model.included_fixations]]).to(DEVICE)

# Get prediction
log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)

# Plot results with less transparent heatmap
f, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Original image with fixations
axs[0].imshow(image)
axs[0].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
axs[0].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
axs[0].set_title('Original Image with Fixations')
axs[0].set_axis_off()

# Heatmap overlaid on image with reduced transparency
axs[1].imshow(image)  # Show the original image first
heatmap = axs[1].imshow(log_density_prediction.detach().cpu().numpy()[0, 0],
                       cmap='jet',
                       alpha=0.5)  # Adjust alpha here (0.5 = 50% opaque)
axs[1].plot(fixation_history_x, fixation_history_y, 'o-', color='red')
axs[1].scatter(fixation_history_x[-1], fixation_history_y[-1], 100, color='yellow', zorder=100)
axs[1].set_title('Saliency Heatmap (Overlay)')
axs[1].set_axis_off()

plt.tight_layout()
plt.show()