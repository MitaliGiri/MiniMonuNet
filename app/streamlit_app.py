import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

# Loading class mappings
with open("class_map.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, len(class_to_idx))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("🏛️ MiniMonuNet: Indian Landmark Finder")
st.write("Upload an image of an Indian monument. The model will try to identify it.")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, 1)
        predicted_class = idx_to_class[top_idx.item()]

    st.success(f"Prediction: **{predicted_class.replace('_', ' ').title()}** ({top_prob.item()*100:.2f}% confidence)")
