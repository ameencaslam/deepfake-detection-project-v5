import streamlit as st
import os
import logging
import numpy as np
from utils_video_processor import VideoProcessor
from pathlib import Path
from utils_image_processor import extract_face, process_image, resize_image_for_display
from utils_model import get_cached_model
from utils_format import format_confidence, format_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_IMAGE_SIZES = {
    "efficientnet": 300,
    "swin": 224
}

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