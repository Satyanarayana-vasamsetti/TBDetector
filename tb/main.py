import streamlit as st
from PIL import Image
import numpy as np
import io
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="TB Detector",
    page_icon="🫁",
    layout="centered"
)

# ---------------------- STYLING ----------------------
st.markdown("""
    <style>
        .centered-img {
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: center;
            gap: 40px;
            margin-top: 20px;
        }
        .label-box {
            font-size: 18px;
            color: #222;
        }
        .label-box span {
            display: block;
            margin-bottom: 10px;
        }
        .result-msg {
            text-align: center;
            font-size: 18px;
            color: green;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>🫁 TB Detection from Chest X-ray</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Upload a chest X-ray to analyze it using AI</p>", unsafe_allow_html=True)

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})
    return YOLO("C:/Users/satya/PycharmProjects/tb/td_detect.pt")

model = load_model()

# ---------------------- SIDEBAR CONFIDENCE ----------------------
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner("🔬 Analyzing X-ray..."):
        results = model.predict(image_np, conf=confidence_threshold)[0]

    if results.boxes and len(results.boxes) > 0:
        # Original image resized for preview
        resized_img = image.resize((400, int(400 * image.size[1] / image.size[0])))
        buf = io.BytesIO()
        resized_img.save(buf, format="JPEG")

        # Extract detection labels and confidence
        labels = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            labels.append(f"🟢 <b>{name}</b> - {conf:.2f}")

        # Display image and prediction info side by side
        st.markdown('<div class="centered-img">', unsafe_allow_html=True)
        st.image(buf.getvalue(), caption="", use_container_width=False)

        st.markdown(
            f"""
            <div class="label-box">
                {"".join(f"<span>{label}</span>" for label in labels)}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Download original image (not annotated)
        st.download_button(
            "📥 Download Uploaded Image",
            data=buf.getvalue(),
            file_name="tb_result.jpg",
            mime="image/jpeg"
        )
    else:
        st.markdown('<p class="result-msg">✅ No signs of Tuberculosis detected.</p>', unsafe_allow_html=True)
