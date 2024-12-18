import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st
from typing import Generator, Tuple, List
import tempfile
import os
from data_handler import get_transforms
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, num_frames: int = 100):
        """
        Initialize video processor
        num_frames: Number of frames to extract from video
        """
        self.num_frames = num_frames
    
    def save_uploaded_video(self, video_file) -> str:
        """Save uploaded video file to temporary location"""
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            return tfile.name
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            raise
    
    def extract_frames(self, video_path: str, progress_callback=None) -> List[np.ndarray]:
        """Extract evenly spaced frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval to get desired number of frames
            frame_interval = max(1, total_frames // self.num_frames)
            
            frames = []
            frame_count = 0
            
            while cap.isOpened() and len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                    if progress_callback:
                        progress = len(frames) / self.num_frames
                        progress_callback(progress)
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def process_video(self, video_path: str, extract_face_fn, process_image_fn, models: List[dict], 
                     progress_callbacks: dict = None) -> Tuple[List[dict], List[List[dict]], List[Image.Image]]:
        """Process video frames and return results"""
        try:
            # Initialize progress callbacks
            if progress_callbacks is None:
                progress_callbacks = {
                    'extract_frames': lambda x: None,
                    'extract_faces': lambda x: None,
                    'process_frames': lambda x: None
                }
            
            # Extract frames
            frames = self.extract_frames(video_path, progress_callbacks['extract_frames'])
            
            # Process frames
            results = []  # Final aggregated results
            frame_results = []  # Results for each frame
            faces = []  # Store detected faces
            processed_frames = 0
            total_frames = len(frames)
            
            for frame in frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Extract face using the same function as images
                face_image, _ = extract_face_fn(frame_pil)
                processed_frames += 1
                progress_callbacks['extract_faces'](processed_frames / total_frames)
                
                if face_image is not None:
                    faces.append(face_image)  # Store the face
                    # Process face through models
                    frame_results_current = []
                    for model_data in models:
                        # Process image using the same function as images
                        processed_image = process_image_fn(face_image, model_data['model_type'])
                        if processed_image is not None:
                            # Make prediction
                            with torch.no_grad():
                                output = model_data['model'](processed_image)
                                probability = torch.sigmoid(output).item()
                                prediction = "FAKE" if probability > 0.5 else "REAL"
                                confidence = probability if prediction == "FAKE" else 1 - probability
                            
                            frame_results_current.append({
                                'model_type': model_data['model_type'],
                                'prediction': prediction,
                                'confidence': confidence
                            })
                    
                    if frame_results_current:
                        frame_results.append(frame_results_current)
                
                progress_callbacks['process_frames'](processed_frames / total_frames)
            
            # Calculate average results across all frames
            final_results = []
            if frame_results:
                # Get unique model types
                model_types = {result['model_type'] for frame_result in frame_results for result in frame_result}
                
                for model_type in model_types:
                    # Get all predictions for this model
                    model_predictions = [
                        result for frame_result in frame_results
                        for result in frame_result
                        if result['model_type'] == model_type
                    ]
                    
                    # Calculate statistics
                    total_predictions = len(model_predictions)
                    fake_count = sum(1 for p in model_predictions if p['prediction'] == "FAKE")
                    avg_confidence = sum(p['confidence'] for p in model_predictions) / total_predictions
                    
                    final_results.append({
                        'model_type': model_type,
                        'prediction': "FAKE" if fake_count > total_predictions/2 else "REAL",
                        'confidence': avg_confidence,
                        'fake_frame_ratio': fake_count / total_predictions
                    })
            
            return final_results, frame_results, faces
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise 