import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define number of classes
num_classes = 38

# Define class names
class_names = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
    "Orange Huanglongbing (Citrus Greening)",
    "Peach Bacterial Spot", "Peach Healthy",
    "Bell Pepper Bacterial Spot", "Bell Pepper Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot", "Tomato Spider Mites", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
]

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("Tomato Disease Prediction(ResNet18).pth", map_location=device))
model.to(device)
model.eval()

# Hooks for Grad-CAM
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# Register hook on the last conv layer
model.layer4[1].conv2.register_forward_hook(forward_hook)
model.layer4[1].conv2.register_backward_hook(backward_hook)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_gradcam(input_tensor, class_idx):
    activations.clear()
    gradients.clear()

    model.zero_grad()
    output = model(input_tensor)
    one_hot = torch.zeros_like(output)
    one_hot[0][class_idx] = 1
    output.backward(gradient=one_hot)

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    act = activations[0][0]
    for i in range(len(pooled_gradients)):
        act[i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(act, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image.convert("RGB"))
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
    return overlay

# Streamlit UI
st.set_page_config(page_title="Tomato Disease Classifier", layout="centered")
st.title("🍅 Tomato Disease Classifier with Grad-CAM")
st.markdown("Upload a tomato leaf image to classify its disease and see the model's focus.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()

    st.success(f"🩺 Predicted class: **{class_names[class_id]}**")

    # Grad-CAM
    heatmap = generate_gradcam(input_tensor, class_id)
    overlay = overlay_heatmap(heatmap, image)
    st.image(overlay, caption="🔥 Grad-CAM Visualization", use_container_width=True)
