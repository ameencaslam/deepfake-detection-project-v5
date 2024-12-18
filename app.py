import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import os
import mlflow
from data_handler import get_transforms
import glob
import logging
import sys
import warnings
import mediapipe as mp
import numpy as np
import cv2
from feature_visualization import get_feature_maps, display_feature_maps
from video_processor import VideoProcessor
import subprocess
import gc
import time
from contextlib import contextmanager

# Configure memory management
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Run setup script if models don't exist
if not os.path.exists("converted_models"):
    try:
        subprocess.run(['bash', 'setup.sh'], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running setup script: {str(e)}")
        st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='timm')
warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.serialization')

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
        /* Control maximum size of all images */
        .stImage > img {
            max-height: 300px !important;
            max-width: 100% !important;
            width: auto !important;
            object-fit: contain;
        }
        
        /* Specific control for face grid images */
        .face-grid-image > img {
            max-height: 200px !important;
            max-width: 100% !important;
            width: auto !important;
            object-fit: contain;
        }
        
        /* Feature map images */
        .feature-map-image > img {
            max-height: 150px !important;
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
        div[data-testid="column"] h4 {
            color: #ffffff;
        }
        div[data-testid="column"] p {
            color: #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_IMAGE_SIZES = {
    "xception": 299,
    "efficientnet": 300,
    "swin": 224,
    "cross_attention": 224,
    "cnn_transformer": 224
}

SESSION_TIMEOUT = 3600  # 1 hour in seconds
LAST_ACTIVITY_KEY = "last_activity"
SESSION_DATA_KEY = "session_data"

@contextmanager
def cleanup_on_exit():
    """Context manager to ensure cleanup when processing is done"""
    try:
        yield
    finally:
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def init_session_state():
    """Initialize or update session state"""
    if LAST_ACTIVITY_KEY not in st.session_state:
        st.session_state[LAST_ACTIVITY_KEY] = time.time()
    if SESSION_DATA_KEY not in st.session_state:
        st.session_state[SESSION_DATA_KEY] = {}
    
    # Update last activity
    st.session_state[LAST_ACTIVITY_KEY] = time.time()

def check_session_timeout():
    """Check if session has timed out"""
    if LAST_ACTIVITY_KEY in st.session_state:
        last_activity = st.session_state[LAST_ACTIVITY_KEY]
        if time.time() - last_activity > SESSION_TIMEOUT:
            # Clear session state
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
        # Initialize model based on type
        model = None
        if model_type == "xception":
            from train_xception import DeepfakeXception
            model = DeepfakeXception()
        elif model_type == "efficientnet":
            from train_efficientnet import DeepfakeEfficientNet
            model = DeepfakeEfficientNet()
        elif model_type == "swin":
            from train_swin import DeepfakeSwin
            model = DeepfakeSwin()
        elif model_type == "cross_attention":
            from train_cross_attention import DeepfakeCrossAttention
            model = DeepfakeCrossAttention()
        elif model_type == "cnn_transformer":
            from train_cnn_transformer import DeepfakeCNNTransformer
            model = DeepfakeCNNTransformer()
            
        if model is None:
            return None
            
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model
    except Exception as e:
        return None

def extract_face(image, padding=0.1):
    """Extract face from image using MediaPipe with padding"""
    # Convert PIL Image to cv2 format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]
    
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5,
        model_selection=1  # Use full range model for better detection
    ) as face_detection:
        # Convert the BGR image to RGB
        results = face_detection.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            # Try with a lower confidence threshold for video frames
            with mp_face_detection.FaceDetection(
                min_detection_confidence=0.3,
                model_selection=1
            ) as face_detection_lower:
                results = face_detection_lower.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                if not results.detections:
                    return None, None
        
        # Get the first face detection
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
        
        # Additional checks for valid dimensions
        if w <= 0 or h <= 0 or x >= width or y >= height:
            return None, None
        
        # Extract face region with padding
        try:
            face_region = img_cv[y:y+h, x:x+w]
            if face_region.size == 0:  # Check if region is empty
                return None, None
                
            # Convert back to RGB for PIL
            face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_region_rgb)
            
            # Create visualization
            img_cv_viz = img_cv.copy()
            cv2.rectangle(img_cv_viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_viz = cv2.cvtColor(img_cv_viz, cv2.COLOR_BGR2RGB)
            
            return face_pil, Image.fromarray(img_viz)
        except Exception as e:
            logger.error(f"Error in face extraction: {str(e)}")
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

def process_image(image, model_type):
    """Process uploaded image using the same transforms as training"""
    try:
        # Get correct image size for model
        img_size = MODEL_IMAGE_SIZES.get(model_type, 224)
        transform = get_transforms(img_size)
        transformed_image = transform(image).unsqueeze(0)
        logger.info(f"Successfully processed image for {model_type} (size: {img_size})")
        return transformed_image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def load_model(model_path, model_type):
    """Load a model using the cached function"""
    return get_cached_model(model_path, model_type)

def format_confidence(confidence):
    """Format confidence score with color based on value"""
    if confidence >= 0.8:
        color = "#00ffff"  # Bright cyan
    elif confidence >= 0.6:
        color = "#00bfff"  # Deep sky blue
    else:
        color = "#87ceeb"  # Sky blue
    return f'<span style="color: {color}; font-weight: bold;">{confidence:.1%}</span>'

def format_prediction(prediction):
    """Format prediction with color (red for FAKE, green for REAL)"""
    color = "#ff4444" if prediction == "FAKE" else "#00ff9d"  # Bright red for FAKE, Bright green for REAL
    return f'<span style="color: {color}; font-weight: bold; font-size: 1.1em;">{prediction}</span>'

def main():
    # Initialize session state
    init_session_state()
    
    # Check for timeout
    if check_session_timeout():
        st.warning("Your session has timed out. Please reload the page.")
        return

    st.title("🔍 Deepfake Detection System")
    st.write("Upload an image or video to check if it's real or fake using multiple deep learning models.")
    
    # Input type selector
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

def process_video_input(video_file):
    try:
        # Store video path in session state
        if 'video_path' not in st.session_state[SESSION_DATA_KEY]:
            st.session_state[SESSION_DATA_KEY]['video_path'] = None
        
        # Get number of frames from user with a start button
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
        
        # Save uploaded video
        video_path = video_processor.save_uploaded_video(video_file)
        
        # Load models with progress tracking
        status_load.text("Loading models: Initializing...")
        progress_load.progress(0.1)
        
        # Load models
        converted_dir = "converted_models"
        if not os.path.exists(converted_dir):
            st.error("Please run convert_models.py first to convert the models!")
            return
        
        model_files = glob.glob(os.path.join(converted_dir, "*_converted.pth"))
        if not model_files:
            st.error("No converted models found! Please run convert_models.py first.")
            return
        
        # Load all models with progress tracking
        models_data = []
        total_models = len(model_files)
        for i, model_path in enumerate(model_files):
            model_type = os.path.basename(model_path).replace("_converted.pth", "")
            status_load.text(f"Loading models: {model_type.upper()}")
            progress_load.progress((i + 1) / (total_models + 1))
            
            model = load_model(model_path, model_type)
            if model is not None:
                models_data.append({
                    'model': model,
                    'model_type': model_type,
                    'image_size': MODEL_IMAGE_SIZES[model_type]
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
        
        # Process video with progress tracking
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
        
        # Clear progress bars and status messages
        progress_load.empty()
        status_load.empty()
        progress_extract.empty()
        status_extract.empty()
        progress_faces.empty()
        status_faces.empty()
        progress_process.empty()
        status_process.empty()
        
        if results:
            # Create tabs for different views
            pred_tab, faces_tab = st.tabs(["Predictions", "Detected Faces"])
            
            with pred_tab:
                st.write("### Model Predictions")
                cols = st.columns(2)
                col_idx = 0
                
                for result in results:
                    with cols[col_idx % 2]:
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                            <h4>{result['model_type'].upper()}</h4>
                            <p>Overall Prediction: {format_prediction(result['prediction'])}<br>
                            Average Confidence: {format_confidence(result['confidence'])}<br>
                            Fake Frames: {result['fake_frame_ratio']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    col_idx += 1
            
            with faces_tab:
                st.write("### Sample Detected Faces")
                # Display a subset of detected faces in a grid
                n_sample_faces = min(12, len(faces))  # Show up to 12 faces
                sample_indices = np.linspace(0, len(faces)-1, n_sample_faces, dtype=int)
                
                # Create grid layout for faces in video processing
                cols = st.columns(4)  # 4 faces per row
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
        if st.session_state[SESSION_DATA_KEY].get('video_path'):
            try:
                os.unlink(st.session_state[SESSION_DATA_KEY]['video_path'])
                st.session_state[SESSION_DATA_KEY]['video_path'] = None
            except:
                pass
        
        clear_session_data()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error processing video: {str(e)}")
        clear_session_data()

def process_image_input(uploaded_file):
    try:
        with cleanup_on_exit():
            # Load and display the image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Extract face
            face_image, viz_image = extract_face(image)
            
            if face_image is None:
                st.error("No face detected in the image. Please upload an image containing a clear face.")
                return
            
            # Create two columns for image and info
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display original image with face detection
                st.write("### Original Image with Face Detection")
                st.image(viz_image, use_container_width=False)
                
                # Display extracted face
                st.write("### Extracted Face")
                display_face = resize_image_for_display(face_image)
                st.image(display_face, use_container_width=False)
                st.write(f"Face size: {face_image.size[0]}x{face_image.size[1]}")
            
            with col2:
                # Load all converted models
                converted_dir = "converted_models"
                if not os.path.exists(converted_dir):
                    st.error("Please run convert_models.py first to convert the models!")
                    return
                
                # Find all converted model files
                model_files = glob.glob(os.path.join(converted_dir, "*_converted.pth"))
                if not model_files:
                    st.error("No converted models found! Please run convert_models.py first.")
                    return
                
                st.write("### Model Predictions")
                
                # Create tabs for predictions and visualizations
                pred_tab, viz_tab = st.tabs(["Predictions", "Feature Visualizations"])
                
                with pred_tab:
                    # Create columns for results
                    cols = st.columns(2)
                    col_idx = 0
                    models_loaded = False
                    
                    progress_bar = st.progress(0)
                    predictions_data = []  # Store predictions for later use
                    
                    for i, model_path in enumerate(model_files):
                        # Get model type from filename
                        model_type = os.path.basename(model_path).replace("_converted.pth", "")
                        
                        with st.spinner(f'Processing with {model_type.upper()}...'):
                            model = load_model(model_path, model_type)
                            if model is None:
                                continue
                            
                            models_loaded = True
                            
                            # Process image for this specific model
                            processed_image = process_image(face_image, model_type)
                            if processed_image is None:
                                st.error(f"Failed to process image for {model_type}")
                                continue
                            
                            # Make prediction
                            try:
                                with torch.no_grad():
                                    output = model(processed_image)
                                    probability = torch.sigmoid(output).item()
                                    prediction = "FAKE" if probability > 0.5 else "REAL"
                                    confidence = probability if prediction == "FAKE" else 1 - probability
                                
                                # Store prediction data
                                predictions_data.append({
                                    'model_type': model_type,
                                    'prediction': prediction,
                                    'confidence': confidence,
                                    'model': model,
                                    'processed_image': processed_image
                                })
                                
                                # Display results in card format
                                with cols[col_idx % 2]:
                                    st.markdown(f"""
                                    <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                                        <h4>{model_type.upper()}</h4>
                                        <p>Prediction: {format_prediction(prediction)}<br>
                                        Confidence: {format_confidence(confidence)}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                col_idx += 1
                                
                            except Exception as e:
                                st.error(f"Error making prediction with {model_type}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(model_files))
                
                with viz_tab:
                    if predictions_data:
                        # Filter for CNN models only
                        cnn_predictions = [p for p in predictions_data 
                                        if p['model_type'].lower() in ['xception', 'efficientnet']]
                        
                        if cnn_predictions:
                            # Add model selector for CNN models only
                            selected_model = st.selectbox(
                                "Select CNN Model for Feature Visualization",
                                options=[p['model_type'].upper() for p in cnn_predictions],
                                format_func=lambda x: f"{x} Model"
                            )
                            
                            # Get selected model data
                            model_data = next(p for p in cnn_predictions if p['model_type'].upper() == selected_model)
                            
                            st.write("### Feature Map Visualization")
                            
                            # Add model-specific descriptions
                            if model_data['model_type'].lower() == 'xception':
                                st.markdown("""
                                <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;'>
                                    <h4 style='color: #00bfff; margin-bottom: 10px;'>🔍 Xception Architecture Analysis</h4>
                                    <p style='color: #e0e0e0; margin-bottom: 5px;'>The model processes features through three main flows:</p>
                                    <ol style='color: #e0e0e0;'>
                                        <li><strong style='color: #00ffff;'>Entry Flow:</strong> Captures basic visual elements like edges, textures, and colors</li>
                                        <li><strong style='color: #00ffff;'>Middle Flow:</strong> Processes intermediate features and patterns</li>
                                        <li><strong style='color: #00ffff;'>Exit Flow:</strong> Combines complex features for final classification</li>
                                    </ol>
                                </div>
                                """, unsafe_allow_html=True)
                            else:  # EfficientNet
                                st.markdown("""
                                <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;'>
                                    <h4 style='color: #00bfff; margin-bottom: 10px;'>🔍 EfficientNet Architecture Analysis</h4>
                                    <p style='color: #e0e0e0; margin-bottom: 5px;'>The model processes features through progressive stages:</p>
                                    <ol style='color: #e0e0e0;'>
                                        <li><strong style='color: #00ffff;'>Initial Stage:</strong> Basic feature extraction (edges, colors)</li>
                                        <li><strong style='color: #00ffff;'>Early Stages (1-2):</strong> Simple pattern recognition</li>
                                        <li><strong style='color: #00ffff;'>Middle Stages (3-4):</strong> Complex pattern processing</li>
                                        <li><strong style='color: #00ffff;'>Late Stages (5-6):</strong> High-level feature composition</li>
                                        <li><strong style='color: #00ffff;'>Final Stage:</strong> Feature refinement for classification</li>
                                    </ol>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; margin-top: 20px;'>
                                <h4 style='color: #00bfff; margin-bottom: 10px;'>🎯 Model's Internal Feature Representations</h4>
                                <p style='color: #e0e0e0;'>
                                    Below are the visualizations of how the model "sees" and processes the image at different stages. 
                                    Brighter areas indicate stronger feature activation, showing what the model considers important for detection.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Get feature maps
                            visualizations = get_feature_maps(
                                model_data['model'],
                                model_data['model_type'],
                                model_data['processed_image']
                            )
                            
                            if visualizations:
                                # Sort visualizations by architectural order
                                if model_data['model_type'].lower() == 'xception':
                                    # Sort by flow order
                                    sorted_maps = {}
                                    for k, v in visualizations.items():
                                        if 'entry_' in k:
                                            sorted_maps[k.replace('entry_', '1_')] = v
                                        elif 'middle_' in k:
                                            sorted_maps[k.replace('middle_', '2_')] = v
                                        elif 'exit_' in k:
                                            sorted_maps[k.replace('exit_', '3_')] = v
                                    
                                    # Display all maps in order
                                    display_feature_maps(face_image, dict(sorted(sorted_maps.items())))
                                
                                else:  # EfficientNet
                                    # Sort by stage order
                                    sorted_maps = {}
                                    stage_order = {
                                        'initial': '1',
                                        'stage1': '2',
                                        'stage2': '3',
                                        'stage3': '4',
                                        'stage4': '5',
                                        'stage5': '6',
                                        'stage6': '7',
                                        'final': '8',
                                        'conv_head': '9'
                                    }
                                    
                                    for k, v in visualizations.items():
                                        for stage, order in stage_order.items():
                                            if stage in k:
                                                sorted_maps[f"{order}_{k}"] = v
                                                break
                                    
                                    # Display all maps in order
                                    display_feature_maps(face_image, dict(sorted(sorted_maps.items())))
                            else:
                                st.info("No feature maps available for this model.")
                        else:
                            st.info("Please select a CNN model (Xception or EfficientNet) for feature visualization.")
                    else:
                        st.warning("No models available for visualization.")
                    
                    if not models_loaded:
                        st.error("No models could be loaded! Please check the model files.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in process_image_input: {str(e)}")
        clear_session_data()

if __name__ == "__main__":
    main()
