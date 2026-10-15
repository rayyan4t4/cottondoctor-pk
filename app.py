import io

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASSES = ["Bacterial Blight", "Curl Virus", "Healthy Leaf", "Herbicide Growth Damage",
           "Leaf Hopper Jassids", "Leaf Redding", "Leaf Variegation"]

ADVICE = {
    "Bacterial Blight": "Angular, water-soaked spots that turn brown. Remove affected leaves, avoid overhead watering, and consider a copper-based bactericide. Rotate crops next season.",
    "Curl Virus": "Leaves curl upward and thicken — spread by whiteflies. Control whiteflies (neem oil or approved insecticide), uproot badly infected plants early, and prefer resistant varieties next sowing.",
    "Healthy Leaf": "No disease detected. Keep scouting your field weekly — early detection is the cheapest protection.",
    "Herbicide Growth Damage": "Twisted, stunted growth from herbicide drift. Avoid spraying on windy days and keep nozzles low. Mildly affected plants often recover on their own.",
    "Leaf Hopper Jassids": "Yellowing leaf edges ('hopper burn') from sap-sucking jassids. Use neem oil or insecticidal soap; encourage natural predators like ladybirds.",
    "Leaf Redding": "Reddening from sucking pests or nutrient stress. Check undersides for jassids/aphids and apply balanced fertiliser as per soil test.",
    "Leaf Variegation": "Yellow-white patches, often viral. Remove severely affected plants, control insect vectors, and do not save seed from infected plants.",
}


@st.cache_resource
def load_model():
    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 7)
    net.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
    net.eval()
    return net


tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.title("CottonDoctor-PK")
st.write("Upload a photo of a cotton leaf — get a disease diagnosis with advice.")

up = st.file_uploader("Cotton leaf photo", type=["jpg", "jpeg", "png"])
if up:
    img = Image.open(io.BytesIO(up.read())).convert("RGB")
    st.image(img, width=320)
    with torch.no_grad():
        probs = torch.softmax(load_model()(tf(img).unsqueeze(0)), 1)[0]
    top = int(probs.argmax())
    st.subheader(f"{CLASSES[top]} — {probs[top]:.1%} confident")
    st.write(ADVICE[CLASSES[top]])
    with st.expander("All class scores"):
        for i, c in enumerate(CLASSES):
            st.write(f"{c}: {probs[i]:.1%}")
    st.caption("AI guidance only — confirm with a local agronomist before treatment. Trained on Bangladesh cotton-leaf images; conditions similar to Punjab/Sindh.")
