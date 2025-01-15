import streamlit as st
import torch
import logging
from utils_eff import DeepfakeEfficientNet
from utils_swin import DeepfakeSwin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_cached_model(model_path, model_type):
    """Cache and share model instances across sessions"""
    try:
        if model_type == "efficientnet":
            model = DeepfakeEfficientNet()
        else:  # swin
            model = DeepfakeSwin()
            
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None