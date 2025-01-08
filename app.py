import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import os
import mlflow
import glob
import logging
import sys
import warnings
import mediapipe as mp
import numpy as np
import cv2
from video_processor import VideoProcessor
import subprocess
import gc
import time
from contextlib import contextmanager
import torchvision.transforms as T
from pathlib import Path

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

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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

# Constants
MODEL_IMAGE_SIZES = {
    "efficientnet": 300,
    "swin": 224
}

def get_transforms(image_size):
    """Get transforms based on model type and image size"""
    if image_size == 300:  # EfficientNet
        transform = T.Compose([
            T.Resize(330),  # Slightly larger for better cropping
            T.CenterCrop(300),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # Swin (224)
        transform = T.Compose([
            T.Resize(256),  # Resize to slightly larger
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

SESSION_TIMEOUT = 3600  # 1 hour in seconds
LAST_ACTIVITY_KEY = "last_activity"
SESSION_DATA_KEY = "session_data"

@contextmanager
def cleanup_on_exit():
    """Context manager to ensure cleanup when processing is done"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def init_session_state():
    """Initialize or update session state"""
    if LAST_ACTIVITY_KEY not in st.session_state:
        st.session_state[LAST_ACTIVITY_KEY] = time.time()
    if SESSION_DATA_KEY not in st.session_state:
        st.session_state[SESSION_DATA_KEY] = {}
    
    st.session_state[LAST_ACTIVITY_KEY] = time.time()

def check_session_timeout():
    """Check if session has timed out"""
    if LAST_ACTIVITY_KEY in st.session_state:
        last_activity = st.session_state[LAST_ACTIVITY_KEY]
        if time.time() - last_activity > SESSION_TIMEOUT:
            st.session_state[SESSION_DATA_KEY] = {}
            st.session_state[LAST_ACTIVITY_KEY] = time.time()
            return True
    return False

def clear_session_data():
    """Clear session-specific data"""
    if SESSION_DATA_KEY in st.session_state:
        st.session_state[SESSION_DATA_KEY] = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@st.cache_resource
def get_cached_model(model_path, model_type):
    """Cache and share model instances across sessions"""
    try:
        if model_type == "efficientnet":
            from train_efficientnet import DeepfakeEfficientNet
            model = DeepfakeEfficientNet()
        else:  # swin
            from train_swin import DeepfakeSwin
            model = DeepfakeSwin()
            
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model_type):
    """Process uploaded image using the same transforms as training"""
    try:
        if image is None:
            logger.error("Received None image in process_image")
            return None
            
        # Get correct image size for model
        img_size = MODEL_IMAGE_SIZES.get(model_type, 224)  # Default to 224 if not found
        
        # Get transforms from data_handler
        transform = get_transforms(img_size)
        
        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            logger.error(f"Expected PIL Image, got {type(image)}")
            return None
        
        # Apply transforms and add batch dimension
        transformed_image = transform(image)
        if transformed_image is None:
            logger.error("Transform returned None")
            return None
            
        transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
        
        logger.info(f"Successfully processed image for {model_type} (size: {img_size})")
        return transformed_image
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def extract_face(image, padding=0.1):
    """Extract face from image using MediaPipe with multiple detection attempts"""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]
    
    # Detection configurations to try
    configs = [
        {"confidence": 0.5, "model": 1},  # Default: high confidence, full range
        {"confidence": 0.5, "model": 0},  # Try short range model
        {"confidence": 0.3, "model": 1},  # Lower confidence, full range
        {"confidence": 0.3, "model": 0},  # Lower confidence, short range
        {"confidence": 0.1, "model": 1},  # Lowest confidence, last resort
    ]
    
    for config in configs:
        with mp_face_detection.FaceDetection(
            min_detection_confidence=config["confidence"],
            model_selection=config["model"]
        ) as face_detection:
            results = face_detection.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # Calculate padding with bounds checking
                pad_w = max(int(bbox.width * width * padding), 0)
                pad_h = max(int(bbox.height * height * padding), 0)
                
                # Convert relative coordinates to absolute with padding
                x = max(0, int(bbox.xmin * width) - pad_w)
                y = max(0, int(bbox.ymin * height) - pad_h)
                w = min(int(bbox.width * width) + (2 * pad_w), width - x)
                h = min(int(bbox.height * height) + (2 * pad_h), height - y)
                
                if w <= 0 or h <= 0 or x >= width or y >= height:
                    continue
                
                try:
                    face_region = img_cv[y:y+h, x:x+w]
                    if face_region.size == 0:
                        continue
                    
                    face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_region_rgb)
                    
                    img_cv_viz = img_cv.copy()
                    cv2.rectangle(img_cv_viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    img_viz = cv2.cvtColor(img_cv_viz, cv2.COLOR_BGR2RGB)
                    
                    logger.info(f"Face detected with confidence {config['confidence']}, model {config['model']}")
                    return face_pil, Image.fromarray(img_viz)
                    
                except Exception as e:
                    logger.error(f"Error extracting face region: {str(e)}")
                    continue
    
    logger.warning("No face detected after trying all configurations")
    return None, None

def resize_image_for_display(image, max_size=300):
    """Resize image for display while maintaining aspect ratio"""
    width, height = image.size
    if width > height:
        if width > max_size:
            ratio = max_size / width
            new_size = (max_size, int(height * ratio))
    else:
        if height > max_size:
            ratio = max_size / height
            new_size = (int(width * ratio), max_size)
    
    if width > max_size or height > max_size:
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def format_confidence(confidence):
    """Format confidence score with color based on value"""
    if confidence >= 0.8:
        color = "#ff4444"  # Strong red for high confidence fake
    elif confidence >= 0.6:
        color = "#ff8c00"  # Orange for medium confidence
    else:
        color = "#00ff9d"  # Green for low confidence (likely real)
    return f'<span style="color: {color}; font-weight: bold;">{confidence:.1%}</span>'

def format_prediction(prediction):
    """Format prediction with color"""
    color = "#ff4444" if prediction == "FAKE" else "#00ff9d"
    return f'<span style="color: {color}; font-weight: bold;">{prediction}</span>'

def process_video_input(video_file):
    try:
        video_path = None  # Initialize video_path to None
        # Initialize session state for processing
        if 'processing_started' not in st.session_state:
            st.session_state.processing_started = False
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'faces' not in st.session_state:
            st.session_state.faces = None

        # Only show the slider and button if processing hasn't started
        if not st.session_state.processing_started:
            # Get number of frames from user
            col1, col2 = st.columns([3, 1])
            with col1:
                num_frames = st.slider("Number of frames to analyze", min_value=10, max_value=300, value=30, step=10,
                                    help="More frames = more accurate but slower processing")
            with col2:
                if st.button("Start Processing", type="primary"):
                    st.session_state.processing_started = True
                    st.session_state.num_frames = num_frames
                    st.rerun()
            return

        # If processing has started but not complete, show the progress bar
        if st.session_state.processing_started and not st.session_state.processing_complete:
            # Create a single progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Initialize video processor
            video_processor = VideoProcessor(num_frames=st.session_state.num_frames)
            video_path = video_processor.save_uploaded_video(video_file)
            
            # Load models
            status_text.text("🔄 Loading models...")
            progress_bar.progress(0.2)
            
            model_dir = Path("runs/models")
            models_data = []
            
            # Load models (EfficientNet and Swin)
            efficientnet_model = get_cached_model(model_dir / "efficientnet/best_model_cpu.pth", "efficientnet")
            if efficientnet_model is not None:
                models_data.append({
                    'model': efficientnet_model,
                    'model_type': 'efficientnet',
                    'image_size': MODEL_IMAGE_SIZES['efficientnet']
                })
            
            swin_model = get_cached_model(model_dir / "swin/best_model_cpu.pth", "swin")
            if swin_model is not None:
                models_data.append({
                    'model': swin_model,
                    'model_type': 'swin',
                    'image_size': MODEL_IMAGE_SIZES['swin']
                })
            
            if not models_data:
                st.error("No models could be loaded! Please check the model files.")
                return
            
            # Define progress callback
            total_progress = {'value': 0}  # Use dictionary to maintain state
            
            def update_progress(stage):
                def callback(progress):
                    # Update total progress based on stage
                    if stage == 'extract_frames':
                        total_progress['value'] = progress * 0.5  # First half
                    elif stage in ['extract_faces', 'process_frames']:
                        total_progress['value'] = 0.5 + (progress * 0.5)  # Second half
                    
                    # Ensure progress only moves forward
                    progress_bar.progress(total_progress['value'])
                    status_text.text(f"🎥 Processing video... {int(total_progress['value'] * 100)}%")
                return callback
            
            progress_callbacks = {
                'extract_frames': update_progress('extract_frames'),
                'extract_faces': update_progress('extract_faces'),
                'process_frames': update_progress('process_frames')
            }
            
            # Process video
            results, frame_results, faces = video_processor.process_video(
                video_path,
                extract_face_fn=extract_face,
                process_image_fn=process_image,
                models=models_data,
                progress_callbacks=progress_callbacks
            )
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.faces = faces
            
            # Clear progress indicators
            progress_container.empty()
            st.session_state.processing_complete = True
            st.rerun()

        # If processing is complete, show the results
        if st.session_state.processing_complete:
            results = st.session_state.results
            faces = st.session_state.faces

            if results:
                st.write("### Model Predictions")
                cols = st.columns(2)
                
                overall_prediction = "REAL"  # Default to REAL
                for idx, result in enumerate(results):
                    with cols[idx % 2]:
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                            <h4>{result['model_type'].upper()}</h4>
                            <p>Overall Prediction: {format_prediction(result['prediction'])}<br>
                            Confidence: {format_confidence(result['confidence'])}<br>
                            Fake Frames: {result['fake_frame_ratio']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Determine overall prediction
                        if result['prediction'] == "FAKE":
                            overall_prediction = "FAKE"
                
                # Display overall verdict with improved styling
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
                
                # Display detected faces right under the results
                st.write("### Sample Detected Faces")
                n_sample_faces = min(12, len(faces))
                sample_indices = np.linspace(0, len(faces)-1, n_sample_faces, dtype=int)
                
                cols = st.columns(4)
                for idx, face_idx in enumerate(sample_indices):
                    with cols[idx % 4]:
                        face = faces[face_idx]
                        resized_face = resize_image_for_display(face, max_size=200)
                        st.markdown('<div class="face-grid-image">', unsafe_allow_html=True)
                        st.image(resized_face, caption=f"Frame {face_idx}", use_container_width=False)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No faces could be detected in the video frames.")
            
            # Cleanup
            if video_path is not None and os.path.exists(video_path):
                os.remove(video_path)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error processing video: {str(e)}")

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
        input_type = st.radio(
            "Select media type to analyze:",
            ["Image", "Video", "Live Camera"],
            horizontal=True,
            label_visibility="visible"
        )
        
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

def is_running_locally():
    """Check if the app is running on localhost."""
    # Streamlit Cloud sets the environment variable 'STREAMLIT_SERVER_PORT'
    return 'STREAMLIT_SERVER_PORT' not in os.environ

def show_live_camera_page():
    """Display the live camera page"""
    st.write("## Live Camera Deepfake Detection")

    # Check if running locally
    if not is_running_locally():
        st.warning("""
            **Live camera feature is only available when running the app locally.**  
            To use this feature, please clone the repository and run the app on your local machine.  
            [GitHub Repository Link](#)
        """)
        return

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
        # Try to open the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access the camera. Please ensure a webcam is connected.")
            st.session_state.camera_active = False
            st.rerun()

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
            st.session_state.current_page = 'live_camera'
            st.rerun()
    elif st.session_state.current_page == 'results':
        with cleanup_on_exit():
            if st.session_state.input_type == "Image":
                process_image_input(st.session_state.uploaded_file)
            else:
                process_video_input(st.session_state.uploaded_file)
    elif st.session_state.current_page == 'live_camera':
        show_live_camera_page()

if __name__ == "__main__":
    main()