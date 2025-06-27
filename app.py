import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load YOLOv8 model
model = YOLO("mask_detector.pt")  # Make sure this file is in the same directory

st.set_page_config(page_title="Face Mask Detector", layout="centered")

st.title("🧠 Real-Time Face Mask Detector")
st.write("Detects: **With Mask**, **Without Mask**, **Mask Worn Incorrectly**")

# Sidebar confidence threshold
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

# Webcam transformer
class YOLOWebcam(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=conf_threshold)
        annotated = results[0].plot()
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# Webcam section
st.subheader("📷 Live Webcam Detection")
webrtc_streamer(
    key="webcam",
    video_transformer_factory=YOLOWebcam,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 320},
            "height": {"ideal": 240}
        },
        "audio": False
    }
)


# Image upload fallback
st.subheader("🖼️ Or Upload an Image")
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        results = model.predict(tmp.name, conf=conf_threshold)
        result_img = results[0].plot()
        rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption="Detections", use_column_width=True)

        st.subheader("Detections:")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"- **{model.names[cls]}**: {conf:.2f}")
