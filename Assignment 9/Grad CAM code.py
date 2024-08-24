import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Grad-CAM specific class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None
        
        # Hook to get gradients of the target layer
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activation = output
    
    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = torch.argmax(output)
        
        # Backward pass
        self.model.zero_grad()
        output[:, target_class].backward()

        # Compute the Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(pooled_gradients.size(0)):
            self.activation[:, i, :, :] *= pooled_gradients[i]
        
        cam = torch.mean(self.activation, dim=1).squeeze().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.size(2), input_image.size(3)))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load an example image
image_path = "C:/Users/dhava/OneDrive/Desktop/LUNGS.jpg"
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)

# Apply Grad-CAM
target_layer = model.layer4[2].conv3  # Target the last convolutional layer of ResNet
grad_cam = GradCAM(model, target_layer)
cam = grad_cam.generate_cam(input_tensor)

# Visualize the Grad-CAM
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
overlay = heatmap + np.array(image.resize((224, 224)), dtype=np.float32) / 255
overlay = overlay / np.max(overlay)

# Display the original image and Grad-CAM result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(overlay)
plt.axis('off')

plt.show()
