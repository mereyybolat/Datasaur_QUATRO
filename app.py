import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ------------------------
# CONFIG
# ------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {
    1: "stop",
    2: "trafficlight",
    3: "speedlimit",
    4: "crosswalk"
}

MODEL_PATH = "cv_model.pth"

# ------------------------
# LOAD MODEL
# ------------------------

@st.cache_resource
def load_model():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

# ------------------------
# INFERENCE FUNCTION
# ------------------------

def detect_image(image):
    transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    return predictions


# ------------------------
# STREAMLIT UI
# ------------------------

st.title("🚦 Road Sign Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    predictions = detect_image(image)

    fig, ax = plt.subplots()
    ax.imshow(image)

    detected_labels = []

    for box, label, score in zip(
        predictions["boxes"],
        predictions["labels"],
        predictions["scores"]
    ):
        if score > 0.4:
            x1, y1, x2, y2 = box.tolist()

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)

            label_name = LABELS.get(label.item(), "unknown")
            detected_labels.append(label_name)

            ax.text(x1, y1 - 5, label_name, color="red", fontsize=12)

    st.pyplot(fig)

    if detected_labels:
        st.success(f"Detected: {', '.join(detected_labels)}")
    else:
        st.warning("No road signs detected.")
