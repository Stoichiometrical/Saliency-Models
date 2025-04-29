import pysaliency
import numpy as np
import matplotlib.pyplot as plt

# Load pretrained SALICON model
saliency_model = pysaliency.load_salicon()

ad_image = "Audi_CR08545138718757879809_v1_image.png"
# Convert ad image to numpy array
image_np = np.array(ad_image)

# Generate heatmap
saliency_map = saliency_model.saliency_map(image_np)

# Display
plt.imshow(saliency_map, cmap='hot')
plt.title("Attention Heatmap")
plt.axis('off')
plt.show()
