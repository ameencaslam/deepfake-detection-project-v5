import streamlit as st
import torch
import os
import logging
import warnings
import subprocess
import json
from pathlib import Path
from utils_video_input import process_video_input
from utils_live_cam import show_live_camera_page
from utils_session import *
from utils_image_input import process_image_input

# Configure memory management
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def ensure_kaggle_credentials():
    """Ensure Kaggle credentials exist for Kaggle CLI.
    Tries env vars first; if missing, writes ~/.kaggle/kaggle.json from st.secrets when available.
    """
    has_env = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if has_env:
        return True

    # Try to create kaggle.json from Streamlit secrets
    try:
        username = st.secrets.get("KAGGLE_USERNAME") if hasattr(st, "secrets") else None
        key = st.secrets.get("KAGGLE_KEY") if hasattr(st, "secrets") else None
    except Exception:
        username, key = None, None

    if username and key:
        # Export env vars for Kaggle CLI and write kaggle.json for redundancy
        os.environ["KAGGLE_USERNAME"] = str(username)
        os.environ["KAGGLE_KEY"] = str(key)
        try:
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            with kaggle_json.open('w') as f:
                json.dump({"username": str(username), "key": str(key)}, f)
        except Exception as e:
            st.warning(f"Could not write Kaggle credentials file: {e}")

    # Return whether we have credentials available one way or another
    return bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY")) or kaggle_json.exists()

# Run setup script if models don't exist
# You can download the model from kaggle 
# For streamlit web deployement, remove if using in localhost or uncomment the line below
#os.makedirs("runs", exist_ok=True)
if not os.path.exists("runs"):
    st.info("Initializing models (first run). Preparing Kaggle credentials...")
    creds_ok = ensure_kaggle_credentials()
    # Sanity check booleans (no secrets printed)
    try:
        st.write(f"Secrets present: {bool(hasattr(st, 'secrets') and st.secrets.get('KAGGLE_USERNAME') and st.secrets.get('KAGGLE_KEY'))}")
        creds_path = Path.home() / ".kaggle" / "kaggle.json"
        st.write(f"Kaggle credentials file present: {creds_path.exists()}")
        st.write(f"Env vars present: {bool(os.getenv('KAGGLE_USERNAME')) and bool(os.getenv('KAGGLE_KEY'))}")
    except Exception:
        pass

    if creds_ok:
        try:
            project_dir = Path(__file__).parent
            st.info("Downloading model weights from Kaggle (this may take a few minutes)...")
            subprocess.run(['bash', 'setup.sh'], check=True, cwd=str(project_dir))
            st.success("Models downloaded and extracted.")
        except subprocess.CalledProcessError as e:
            st.error(f"Error running setup script: {str(e)}")
            # Do not stop the app; let UI load so users see guidance
    else:
        st.warning("Kaggle credentials not detected. Set KAGGLE_USERNAME and KAGGLE_KEY in Streamlit Secrets to enable automatic model download, or upload models to runs/models/... manually.")

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