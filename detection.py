import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import tempfile
import cv2
import os
import time

# Page config
st.set_page_config(page_title="🎯 Object Detection (Image & Video)", layout="centered")
st.title("🎯 Object Detection using YOLOv8")

# Model path
MODEL_PATH = r"C:\Users\somes\Desktop\folder\yolov8_final.pt"

# Load the model
try:
    model = YOLO(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Selection: Image or Video
option = st.radio("Choose input type:", ["Image", "Video"], horizontal=True)

# -------------- IMAGE DETECTION --------------
if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("🔍 Detecting objects in image..."):
                results = model.predict(image, conf=0.25, device='cpu')
                result_image = Image.fromarray(results[0].plot())

            st.image(result_image, caption="Detected Image", use_container_width=True)
            st.success("✅ Detection complete!")

            detections = results[0].boxes
            st.write(f"Number of detections: {len(detections)}")
            for box in detections:
                class_name = results[0].names[int(box.cls)]
                confidence = float(box.conf)
                st.write(f"Detected: {class_name} (Confidence: {confidence:.2f})")

        except Exception as e:
            st.error(f"❌ Error during image detection: {str(e)}")

# -------------- VIDEO DETECTION --------------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)
        process_button = st.button("🔍 Process Video")

        if process_button:
            st.info("⏳ Processing video, please wait...")

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=0.25, device='cpu')
                annotated_frame = results[0].plot()
                annotated_frame = annotated_frame.astype(np.uint8)  # Ensure proper format
                out.write(annotated_frame)
                frame_count += 1

            cap.release()
            out.release()

            # Wait for the file to be properly saved
            time.sleep(1)

            st.success(f"✅ Processed {frame_count} frames!")
            st.write("Processed video saved at:", output_path)

            with open(output_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)

# -------------- SIDEBAR INFO --------------
st.sidebar.header("About")
st.sidebar.info("""
This app supports both image and video uploads for object detection using a YOLOv8 model trained on custom data.

1. Choose input type (image or video).
2. Upload your file.
3. Let the model do the rest!
""")
