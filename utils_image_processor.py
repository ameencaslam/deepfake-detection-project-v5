from PIL import Image
import logging
import numpy as np
import cv2
import mediapipe as mp
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def process_image(image, model_type):
    """Process uploaded image using the same transforms as training"""
    try:
        if image is None:
            logger.error("Received None image in process_image")
            return None
            
        # Get correct image size for model
        img_size = MODEL_IMAGE_SIZES.get(model_type, 224)  # Default to 224 if not found
        
        # Get transforms
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