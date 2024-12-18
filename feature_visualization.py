import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import streamlit as st
from train_xception import DeepfakeXception
from train_efficientnet import DeepfakeEfficientNet
from train_swin import DeepfakeSwin
from train_cross_attention import DeepfakeCrossAttention
from train_cnn_transformer import DeepfakeCNNTransformer

class FeatureExtractor:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        try:
            if self.model_type == "xception":
                # Entry Flow
                # Stem
                if hasattr(self.model.backbone, 'conv1'):
                    self.hooks.append(
                        self.model.backbone.conv1.register_forward_hook(
                            lambda m, i, o: self._hook_fn('entry_stem_conv1', o)
                        )
                    )
                
                # Entry flow blocks
                entry_blocks = ['block1', 'block2', 'block3']
                for block_name in entry_blocks:
                    if hasattr(self.model.backbone, block_name):
                        block = getattr(self.model.backbone, block_name)
                        # Hook for separable convolutions
                        self.hooks.append(
                            block.rep.register_forward_hook(
                                lambda m, i, o, name=block_name: self._hook_fn(f'entry_{name}_sepconv', o)
                            )
                        )
                        # Hook for residual connection if exists
                        if hasattr(block, 'skip'):
                            self.hooks.append(
                                block.skip.register_forward_hook(
                                    lambda m, i, o, name=block_name: self._hook_fn(f'entry_{name}_residual', o)
                                )
                            )
                
                # Middle Flow (8 identical blocks)
                middle_blocks = [f'block{i}' for i in range(4, 12)]  # Changed to capture all 8 blocks
                for block_name in middle_blocks:
                    if hasattr(self.model.backbone, block_name):
                        block = getattr(self.model.backbone, block_name)
                        self.hooks.append(
                            block.rep.register_forward_hook(
                                lambda m, i, o, name=block_name: self._hook_fn(f'middle_{name}_sepconv', o)
                            )
                        )
                
                # Exit Flow
                exit_blocks = ['block12', 'block13']  # Last two blocks
                for block_name in exit_blocks:
                    if hasattr(self.model.backbone, block_name):
                        block = getattr(self.model.backbone, block_name)
                        # Hook for separable convolutions
                        self.hooks.append(
                            block.rep.register_forward_hook(
                                lambda m, i, o, name=block_name: self._hook_fn(f'exit_{name}_sepconv', o)
                            )
                        )
                        # Hook for final convolutions
                        if hasattr(block, 'conv3'):
                            self.hooks.append(
                                block.conv3.register_forward_hook(
                                    lambda m, i, o, name=block_name: self._hook_fn(f'exit_{name}_conv3', o)
                                )
                            )
                
            elif self.model_type == "efficientnet":
                # Initial conv layer
                if hasattr(self.model.backbone, '_conv_stem'):
                    self.hooks.append(
                        self.model.backbone._conv_stem.register_forward_hook(
                            lambda m, i, o: self._hook_fn('initial_conv_stem', o)
                        )
                    )
                
                # MBConv blocks
                if hasattr(self.model.backbone, '_blocks'):
                    current_stage = 0
                    prev_channels = None
                    
                    for idx, block in enumerate(self.model.backbone._blocks):
                        # Detect stage changes based on channel changes or expansion ratio changes
                        current_channels = block._project_conv.out_channels
                        if prev_channels != current_channels:
                            current_stage += 1
                            prev_channels = current_channels
                        
                        # Register hooks for key components of MBConv block
                        # Expansion conv
                        if hasattr(block, '_expand_conv'):
                            self.hooks.append(
                                block._expand_conv.register_forward_hook(
                                    lambda m, i, o, s=current_stage, idx=idx: 
                                    self._hook_fn(f'stage{s}_block{idx}_expand', o)
                                )
                            )
                        
                        # Depthwise conv
                        if hasattr(block, '_depthwise_conv'):
                            self.hooks.append(
                                block._depthwise_conv.register_forward_hook(
                                    lambda m, i, o, s=current_stage, idx=idx: 
                                    self._hook_fn(f'stage{s}_block{idx}_depthwise', o)
                                )
                            )
                        
                        # SE block if exists
                        if hasattr(block, '_se_reduce'):
                            self.hooks.append(
                                block._se_reduce.register_forward_hook(
                                    lambda m, i, o, s=current_stage, idx=idx: 
                                    self._hook_fn(f'stage{s}_block{idx}_se', o)
                                )
                            )
                        
                        # Project conv
                        if hasattr(block, '_project_conv'):
                            self.hooks.append(
                                block._project_conv.register_forward_hook(
                                    lambda m, i, o, s=current_stage, idx=idx: 
                                    self._hook_fn(f'stage{s}_block{idx}_project', o)
                                )
                            )
                
                # Final conv layer
                if hasattr(self.model.backbone, '_conv_head'):
                    self.hooks.append(
                        self.model.backbone._conv_head.register_forward_hook(
                            lambda m, i, o: self._hook_fn('final_conv_head', o)
                        )
                    )
            
            else:
                logger.info(f"Feature extraction not supported for model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Error registering hooks for {self.model_type}: {str(e)}")
            raise
    
    def _hook_fn(self, layer_name, output):
        self.features[layer_name] = output.detach()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_features(self):
        return self.features

class FeatureVisualizer:
    @staticmethod
    def normalize_feature_map(feature_map):
        if len(feature_map.shape) == 4:
            feature_map = feature_map.squeeze(0)
        if len(feature_map.shape) == 3:
            # Use max activation across channels for better visualization
            feature_map = feature_map.max(0)[0]
        
        feature_map = feature_map.cpu().numpy()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        return feature_map
    
    @staticmethod
    def create_heatmap(feature_map, size=(224, 224)):
        # Resize feature map
        feature_map_resized = cv2.resize(feature_map, size)
        
        # Apply colormap (using COLORMAP_INFERNO for better visibility)
        heatmap = cv2.applyColorMap(np.uint8(255 * feature_map_resized), cv2.COLORMAP_INFERNO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap
    
    @staticmethod
    def overlay_heatmap(image, heatmap, alpha=0.7):
        # Convert PIL Image to numpy array if necessary
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize image to match heatmap
        image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
        
        # Create overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay

def get_feature_maps(model, model_type, image_tensor):
    """Extract and visualize feature maps for a given model and image"""
    # Initialize feature extractor
    extractor = FeatureExtractor(model, model_type)
    visualizer = FeatureVisualizer()
    
    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Get features
    features = extractor.get_features()
    
    # Process features based on model type
    visualizations = {}
    
    if model_type == "xception":
        for layer_name, feature in features.items():
            feature_map = visualizer.normalize_feature_map(feature)
            heatmap = visualizer.create_heatmap(feature_map)
            visualizations[f"Xception {layer_name}"] = heatmap
            
    elif model_type == "efficientnet":
        for layer_name, feature in features.items():
            feature_map = visualizer.normalize_feature_map(feature)
            heatmap = visualizer.create_heatmap(feature_map)
            visualizations[f"EfficientNet {layer_name}"] = heatmap
            
    elif model_type == "swin":
        for layer_name, feature in features.items():
            if "layer" in layer_name:
                feature_map = visualizer.normalize_feature_map(feature)
                heatmap = visualizer.create_heatmap(feature_map)
                visualizations[f"Swin {layer_name}"] = heatmap
                
    elif model_type == "cross_attention":
        for layer_name, feature in features.items():
            attn_map = visualizer.visualize_attention(feature)
            visualizations[f"Cross Attention {layer_name}"] = attn_map
            
    elif model_type == "cnn_transformer":
        for layer_name, feature in features.items():
            if "cnn" in layer_name:
                feature_map = visualizer.normalize_feature_map(feature)
                heatmap = visualizer.create_heatmap(feature_map)
            else:
                heatmap = visualizer.visualize_attention(feature)
            visualizations[f"CNN-Transformer {layer_name}"] = heatmap
    
    # Remove hooks
    extractor.remove_hooks()
    
    return visualizations

def display_feature_maps(image, visualizations):
    """Display feature maps in a simple grid layout with overlay control"""
    if not visualizations:
        st.warning("No feature maps available to display.")
        return
    
    # Convert image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create a more organized layout
    st.write("### Model's Internal Feature Representations")
    
    # Display options
    overlay_alpha = st.slider("Overlay Intensity", 0.0, 1.0, 0.7, 0.1)
    
    # Calculate grid dimensions
    n_maps = len(visualizations)
    n_cols = 4  # Show 4 columns
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    # Create grid layout
    for i in range(0, n_maps, n_cols):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = i + j
            if idx < n_maps:
                name = list(visualizations.keys())[idx]
                heatmap = list(visualizations.values())[idx]
                
                # Create overlay
                overlay = FeatureVisualizer.overlay_heatmap(image, heatmap, overlay_alpha)
                
                # Display in grid
                with cols[j]:
                    st.image(overlay, caption=name.replace('_', ' ').title())

def get_feature_description(feature_name):
    """Get description for different types of features"""
    descriptions = {
        'entry_stem': "Initial feature extraction that captures basic visual elements like edges and textures.",
        'entry_block': "Early processing that starts to combine basic features into more complex patterns.",
        'middle_block': "Deep feature processing that identifies higher-level patterns and structures.",
        'exit_block': "Final feature refinement that combines complex patterns for classification.",
        'stage': "Progressive feature transformation in the network.",
        'expand': "Channel expansion to increase feature representation capacity.",
        'depthwise': "Spatial feature processing that captures local patterns.",
        'se': "Channel attention that highlights important feature channels.",
        'project': "Feature dimension reduction and refinement.",
        'conv_head': "Final feature processing before classification."
    }
    
    for key, desc in descriptions.items():
        if key in feature_name.lower():
            return desc
    return ""  # Return empty string instead of default text