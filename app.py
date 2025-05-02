
'''
 uvicorn app:app --reload
'''

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

# Static and templates
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load ResNet-50 model
model = models.resnet50(pretrained=True)  # Using ResNet-50
model.fc = nn.Linear(2048, 2)  # ResNet-50 has 2048 features from the final fully connected layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

target_layer = model.layer4[-1]  # Target layer for GradCAM (last layer of ResNet-50)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        self.target_layer.register_full_backward_hook(self.backward_hook)

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

def classify_and_process(image_path):
    image_tensor = preprocess_image(image_path)
    output = model(image_tensor)
    prediction = output.argmax().item()

    if prediction == 0:
        return "normal", "Prediction: Normal"
    else:
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

        cv2.imwrite("static/outputs/overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return "oscc", "Prediction: OSCC"

# Home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Upload route
@app.post("/upload", response_class=HTMLResponse)
def upload(request: Request, file: UploadFile = File(...)):
    upload_path = os.path.join("static", "uploads", file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, message = classify_and_process(upload_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction,
        "message": message,
        "image_path": f"/static/uploads/{file.filename}",
        "overlay_path": "/static/outputs/overlay.jpg" if prediction == "oscc" else None
    })
