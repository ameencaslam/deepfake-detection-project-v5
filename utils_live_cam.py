import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from utils_model import get_cached_model
from utils_image_processor import extract_face, process_image

def show_live_camera_page():
    """Display the live camera page"""
    st.write("## Live Camera Deepfake Detection")

    # Initialize session state for camera control
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    # Model selection
    model_type = st.radio(
        "Select model for live detection:",
        ["EfficientNet", "Swin Transformer"],
        horizontal=True
    )

    # Start/Stop button with dynamic text
    if st.button("Stop Camera" if st.session_state.camera_active else "Start Camera"):
        st.session_state.camera_active = not st.session_state.camera_active
        st.rerun()  # Force a rerun to update the UI immediately

    # Status message below the button
    if st.session_state.camera_active:
        st.write("Camera is running. Press 'Stop Camera' to end the feed.")
    else:
        st.write("Camera is stopped. Press 'Start Camera' to begin.")

    # Placeholder for the video feed
    video_placeholder = st.empty()

    # Load the selected model
    model_dir = Path("runs/models")
    if model_type == "EfficientNet":
        model = get_cached_model(model_dir / "efficientnet/best_model_cpu.pth", "efficientnet")
    else:
        model = get_cached_model(model_dir / "swin/best_model_cpu.pth", "swin")

    if model is None:
        st.error("Failed to load the selected model.")
        return

    # Start camera feed
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)  # Open webcam
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break

            # Convert frame to RGB (MediaPipe requires RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces and draw bounding boxes
            face_image, viz_image = extract_face(Image.fromarray(frame_rgb))
            if face_image is not None:
                # Process face with the selected model
                processed_image = process_image(face_image, model_type.lower())
                if processed_image is not None:
                    with torch.no_grad():
                        output = model(processed_image)
                        probability = torch.sigmoid(output).item()
                        prediction = "FAKE" if probability > 0.5 else "REAL"
                        confidence = probability if prediction == "FAKE" else 1 - probability

                    # Set color based on prediction
                    if prediction == "FAKE":
                        text_color = (0, 0, 255)  # Red for FAKE
                    else:
                        text_color = (0, 255, 0)  # Green for REAL

                # Draw the face detection square on the frame
                if viz_image is not None:
                    # Convert viz_image (with face detection square) back to OpenCV format
                    frame_with_square = cv2.cvtColor(np.array(viz_image), cv2.COLOR_RGB2BGR)
                    # Overlay the predictions on the frame with the face detection square
                    cv2.putText(frame_with_square, f"Prediction: {prediction}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    cv2.putText(frame_with_square, f"Confidence: {confidence*100:.0f}%", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    # Use the frame with both the square and predictions
                    frame = frame_with_square

            # Display the frame with a specific width
            video_placeholder.image(frame, channels="BGR", width=800)  # Set width to 800 pixels

        # Release the camera when stopped
        cap.release()
        st.session_state.camera_active = False  # Ensure state is reset
        st.rerun()  # Force a rerun to update the UI