import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os 
os.makedirs("static/outputs", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)


# Load model
model = models.resnet50(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Target layer
target_layer = model.layer4[-1]

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, image, class_idx=None):
        output = self.model(image)
        if class_idx is None:
            class_idx = output.argmax().item()
        self.model.zero_grad()
        output[:, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def get_bounding_boxes(heatmap, threshold=0.6):
    binary_map = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    return boxes

def process_image(image_path):
    image_tensor = preprocess_image(image_path)
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_cam(image_tensor)
    boxes = get_bounding_boxes(heatmap, threshold=0.6)

    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.5, heatmap_colored, 0.5, 0)

    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    os.makedirs("static/outputs", exist_ok=True)
    cv2.imwrite("static/outputs/original.jpg", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite("static/outputs/heatmap.jpg", np.uint8(255 * heatmap))
    cv2.imwrite("static/outputs/overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
