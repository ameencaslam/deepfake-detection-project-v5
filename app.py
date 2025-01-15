import streamlit as st
import torch
from PIL import Image
import os
import logging
import warnings
import subprocess
from pathlib import Path
from utils_image import process_image, extract_face, resize_image_for_display
from utils_video import process_video_input
from utils_live_cam import show_live_camera_page
from utils_model import get_cached_model
from utils_format import format_confidence, format_prediction
from utils_session import *

# Configure memory management
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Run setup script if models don't exist
# You can download the model from kaggle 
# For streamlit web deployement, remove if using in localhost or uncomment the line below
#os.makedirs("runs", exist_ok=True)
if not os.path.exists("runs"):
    try:
        subprocess.run(['bash', 'setup.sh'], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running setup script: {str(e)}")
        st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.serialization')

# Set page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stImage > img {
            max-height: 300px !important;
            max-width: 100% !important;
            width: auto !important;
            object-fit: contain;
        }
        
        .face-grid-image > img {
            max-height: 200px !important;
            max-width: 100% !important;
            width: auto !important;
            object-fit: contain;
        }
        
        div.stMarkdown {
            max-width: 100%;
        }
        div[data-testid="column"] {
            background-color: #1e1e1e;
            padding: 15px;
            border-radius: 10px;
            margin: 5px;
            border: 1px solid #333;
        }
    </style>
""", unsafe_allow_html=True)

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
                            prediction = "FAKE" if probability > 0.5 else "REAL"
                            confidence = probability if prediction == "FAKE" else 1 - probability
                        
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                            <h4>EFFICIENTNET</h4>
                            <p>Prediction: {format_prediction(prediction)}<br>
                            Confidence: {format_confidence(confidence)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to process image for EfficientNet")

            overall_prediction = "REAL"  # Default to REAL
            if prediction == "FAKE":
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
                            prediction = "FAKE" if probability > 0.5 else "REAL"
                            confidence = probability if prediction == "FAKE" else 1 - probability
                        
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                            <h4>SWIN TRANSFORMER</h4>
                            <p>Prediction: {format_prediction(prediction)}<br>
                            Confidence: {format_confidence(confidence)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to process image for Swin")
            
            # Display overall verdict
            if prediction == "FAKE":
                overall_prediction = "FAKE"
            
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
        
def show_home_page():
    """Display the landing/home page"""
    # Custom CSS for the header and input section
    st.markdown("""
        <style>
            /* Header styling */
            .header-container {
                text-align: center;
                padding: 2rem 0;
                margin-bottom: 2rem;
            }
            .main-title {
                color: #ffffff;
                font-size: 2.5rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                color: #a0a0a0;
                font-size: 1.1rem;
                font-weight: 300;
                margin-bottom: 2rem;
            }
            
            /* Input section styling */
            .input-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            /* Radio button styling */
            div[data-testid="stHorizontalBlock"] {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 1rem;
                display: flex;
                justify-content: center;
                gap: 2rem;
                margin-bottom: 1rem;
            }
            
            div.row-widget.stRadio > div {
                flex-direction: row;
                justify-content: center;
                gap: 2rem;
            }
            
            div.row-widget.stRadio > div[role="radiogroup"] > label {
                background: rgba(255, 255, 255, 0.1);
                padding: 0.5rem 2rem;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            
            /* File uploader styling */
            .uploadFile {
                margin-top: 1rem;
                border: 2px dashed rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                background: rgba(255, 255, 255, 0.05);
            }
            
            .upload-text {
                color: #a0a0a0;
                font-size: 0.9rem;
                margin-top: 0.5rem;
                text-align: center;
            }
        </style>
        
        <div class="header-container">
            <h1 class="main-title">Deepfake🔍Detection</h1>
            <p class="subtitle">Analyze images and videos for potential deepfake manipulation</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input container
    with st.container():
        # Input type selection
        input_options = ["Image", "Video", "Live Camera"]
        input_type = st.radio(
            "Select media type to analyze:",
            input_options,
            horizontal=True,
            label_visibility="visible"
        )
        
        # Show a message if "Live Camera" is selected but not running locally
        if input_type == "Live Camera" and not is_running_locally():
            st.info(
                "⚠️ The **Live Camera** feature is only available when running this app locally. "
                "To use this feature, please clone the GitHub repository and run the app on your machine. "
                "[GitHub Repository Link](https://github.com/ameencaslam/deepfake-detection-project-v5)"
            )
            return "Live Camera", None
        
        # File uploader for image and video
        if input_type in ["Image", "Video"]:
            if input_type == "Image":
                uploaded_file = st.file_uploader(
                    "Upload Image",
                    type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed"
                )
                st.markdown(
                    '<p class="upload-text">Supported formats: JPG, JPEG, PNG</p>',
                    unsafe_allow_html=True
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload Video",
                    type=["mp4", "avi", "mov"],
                    label_visibility="collapsed"
                )
                st.markdown(
                    '<p class="upload-text">Supported formats: MP4, AVI, MOV • Max size: 200MB</p>',
                    unsafe_allow_html=True
                )
            return input_type, uploaded_file
        else:  # Live Camera
            return "Live Camera", None

def main():
    init_session_state()

    if check_session_timeout():
        st.warning("Your session has timed out. Please reload the page.")
        return

    # Initialize or get the current page state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

    # Show back button if not on the home page
    if st.session_state.current_page != 'home':
        if st.button('← Back to Home'):
            clear_session_state()
            st.session_state.current_page = 'home'
            st.rerun()

    # Display appropriate page
    if st.session_state.current_page == 'home':
        input_type, uploaded_file = show_home_page()
        if uploaded_file is not None:
            st.session_state.input_type = input_type
            st.session_state.uploaded_file = uploaded_file
            st.session_state.current_page = 'results'
            st.rerun()
        elif input_type == "Live Camera":
            if is_running_locally():
                st.session_state.current_page = 'live_camera'
                st.rerun()
            else:
                # The message is already shown in show_home_page, so no need to handle it here
                pass
    elif st.session_state.current_page == 'results':
        with cleanup_on_exit():
            if st.session_state.input_type == "Image":
                process_image_input(st.session_state.uploaded_file)
            else:
                process_video_input(st.session_state.uploaded_file)
    elif st.session_state.current_page == 'live_camera':
        if is_running_locally():
            show_live_camera_page()
        else:
            st.error("Live Camera option is only available when running locally.")

if __name__ == "__main__":
    main()