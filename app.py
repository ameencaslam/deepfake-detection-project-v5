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
        # Get number of frames from user
        col1, col2 = st.columns([3, 1])
        with col1:
            num_frames = st.slider("Number of frames to analyze", min_value=10, max_value=300, value=100, step=10,
                                help="More frames = more accurate but slower processing")
        with col2:
            start_button = st.button("Start Processing", type="primary")
        
        if not start_button:
            return
        
        # Create progress bars
        st.write("### Processing Video")
        progress_load = st.progress(0)
        status_load = st.empty()
        progress_extract = st.progress(0)
        status_extract = st.empty()
        progress_faces = st.progress(0)
        status_faces = st.empty()
        progress_process = st.progress(0)
        status_process = st.empty()
        
        # Initialize video processor
        video_processor = VideoProcessor(num_frames=num_frames)
        video_path = video_processor.save_uploaded_video(video_file)
        
        # Load models
        status_load.text("Loading models...")
        progress_load.progress(0.2)
        
        model_dir = Path("runs/models")
        models_data = []
        
        # Load EfficientNet
        efficientnet_model = get_cached_model(model_dir / "efficientnet/best_model_cpu.pth", "efficientnet")
        if efficientnet_model is not None:
            models_data.append({
                'model': efficientnet_model,
                'model_type': 'efficientnet',
                'image_size': MODEL_IMAGE_SIZES['efficientnet']
            })
        progress_load.progress(0.6)
        
        # Load Swin
        swin_model = get_cached_model(model_dir / "swin/best_model_cpu.pth", "swin")
        if swin_model is not None:
            models_data.append({
                'model': swin_model,
                'model_type': 'swin',
                'image_size': MODEL_IMAGE_SIZES['swin']
            })
        
        progress_load.progress(1.0)
        status_load.text("Models loaded successfully!")
        
        if not models_data:
            st.error("No models could be loaded! Please check the model files.")
            return
        
        def update_progress(progress_bar, status_placeholder, stage):
            def callback(progress):
                progress_bar.progress(progress)
                status_placeholder.text(f"{stage}: {progress:.1%}")
            return callback
        
        progress_callbacks = {
            'extract_frames': update_progress(progress_extract, status_extract, "Extracting frames"),
            'extract_faces': update_progress(progress_faces, status_faces, "Detecting faces"),
            'process_frames': update_progress(progress_process, status_process, "Processing frames")
        }
        
        results, frame_results, faces = video_processor.process_video(
            video_path,
            extract_face_fn=extract_face,
            process_image_fn=process_image,
            models=models_data,
            progress_callbacks=progress_callbacks
        )
        
        # Clear progress bars
        for progress in [progress_load, progress_extract, progress_faces, progress_process]:
            progress.empty()
        for status in [status_load, status_extract, status_faces, status_process]:
            status.empty()
        
        if results:
            pred_tab, faces_tab = st.tabs(["Predictions", "Detected Faces"])
            
            with pred_tab:
                st.write("### Model Predictions")
                cols = st.columns(2)
                
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
            
            with faces_tab:
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
        if os.path.exists(video_path):
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
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("### Original Image with Face Detection")
                st.image(viz_image, use_container_width=False)
                
                st.write("### Extracted Face")
                display_face = resize_image_for_display(face_image)
                st.image(display_face, use_container_width=False)
                st.write(f"Face size: {face_image.size[0]}x{face_image.size[1]}")
            
            with col2:
                st.write("### Model Predictions")
                
                # Load models
                model_dir = Path("runs/models")
                cols = st.columns(2)
                
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
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in process_image_input: {str(e)}")
        clear_session_data()

def main():
    init_session_state()
    
    if check_session_timeout():
        st.warning("Your session has timed out. Please reload the page.")
        return

    st.title("🔍 Deepfake Detection System")
    st.write("Upload an image or video to check if it's real or fake using multiple deep learning models.")
    
    input_type = st.radio("Select input type:", ["Image", "Video"], horizontal=True)
    
    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            with cleanup_on_exit():
                process_image_input(uploaded_file)
    else:
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            with cleanup_on_exit():
                process_video_input(uploaded_file)

if __name__ == "__main__":
    main()
