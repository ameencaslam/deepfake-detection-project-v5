import streamlit as st
import torch
from PIL import Image
import logging
from pathlib import Path
from utils_image_processor import *
from utils_model import get_cached_model
from utils_format import format_confidence, format_prediction
from utils_session import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image_input(uploaded_file):
    try:
        with cleanup_on_exit():
            image = Image.open(uploaded_file).convert('RGB')
            face_image, viz_image = extract_face(image)
            
            if face_image is None:
                st.error("No face detected in the image. Please upload an image containing a clear face.")
                return
            
            # Model predictions section (moved to the top)
            st.write("### Model Predictions")
            
            # Load models
            model_dir = Path("runs/models")
            cols = st.columns(2)  # Two columns for EfficientNet and Swin predictions
            
            efficientnet_pred = None
            swin_pred = None

            # Process with EfficientNet
            with cols[0]:
                efficientnet_model = get_cached_model(
                    model_dir / "efficientnet/best_model_cpu.pth", 
                    "efficientnet"
                )
                if efficientnet_model is not None:
                    processed_image = process_image(face_image, "efficientnet")
                    if processed_image is not None:
                        with torch.no_grad():
                            output = efficientnet_model(processed_image)
                            probability = torch.sigmoid(output).item()
                            efficientnet_pred = "FAKE" if probability > 0.5 else "REAL"
                            confidence = probability if efficientnet_pred == "FAKE" else 1 - probability
                        
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                            <h4>EFFICIENTNET</h4>
                            <p>Prediction: {format_prediction(efficientnet_pred)}<br>
                            Confidence: {format_confidence(confidence)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to process image for EfficientNet")
                else:
                    st.warning("EfficientNet model not found. Ensure runs/models/efficientnet/best_model_cpu.pth exists.")

            overall_prediction = "REAL"  # Default to REAL
            if efficientnet_pred == "FAKE":
                overall_prediction = "FAKE"
            
            # Process with Swin
            with cols[1]:
                swin_model = get_cached_model(
                    model_dir / "swin/best_model_cpu.pth", 
                    "swin"
                )
                if swin_model is not None:
                    processed_image = process_image(face_image, "swin")
                    if processed_image is not None:
                        with torch.no_grad():
                            output = swin_model(processed_image)
                            probability = torch.sigmoid(output).item()
                            swin_pred = "FAKE" if probability > 0.5 else "REAL"
                            confidence = probability if swin_pred == "FAKE" else 1 - probability
                        
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                            <h4>SWIN TRANSFORMER</h4>
                            <p>Prediction: {format_prediction(swin_pred)}<br>
                            Confidence: {format_confidence(confidence)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to process image for Swin")
                else:
                    st.warning("Swin model not found. Ensure runs/models/swin/best_model_cpu.pth exists.")
            
            # Display overall verdict
            if swin_pred == "FAKE":
                overall_prediction = "FAKE"
            
            if efficientnet_pred is None and swin_pred is None:
                st.error("No models loaded. Upload weights or set Kaggle credentials to download them automatically.")
                return

            st.markdown(f"""
            <div style='
                background-color: {"rgba(255, 68, 68, 0.1)" if overall_prediction == "FAKE" else "rgba(0, 255, 157, 0.1)"};
                border: 3px solid {"#ff4444" if overall_prediction == "FAKE" else "#00ff9d"};
                border-radius: 15px;
                padding: 20px;
                margin: 30px 0;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            '>
                <div style='
                    font-size: 4em;
                    margin: 0;
                    color: {"#ff4444" if overall_prediction == "FAKE" else "#00ff9d"};
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
                    font-weight: bold;
                '>
                    {overall_prediction}
                </div>
                <p style='
                    font-size: 1.2em;
                    margin: 10px 0 0 0;
                    color: {"#ff4444" if overall_prediction == "FAKE" else "#00ff9d"};
                '>
                    Final Verdict
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Image section (moved below predictions)
            st.write("")  # Add some spacing
            
            # Use columns to align the images side by side with a gap
            col1, col2 = st.columns([1, 1])  # Equal width for both columns
            
            with col1:
                # Resize the original image for display
                resized_viz_image = resize_image_for_display(viz_image, max_size=500)  # Adjust max_size as needed
                st.image(resized_viz_image, caption="Original Image with Face Detection")  # Caption under the image
            
            with col2:
                # Resize the extracted face for display
                display_face = resize_image_for_display(face_image, max_size=500)  # Adjust max_size as needed
                st.image(display_face, caption="Extracted Face")  # Caption under the image
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in process_image_input: {str(e)}")
        clear_session_data()