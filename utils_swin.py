import torch
import torch.nn as nn
import timm

class DeepfakeSwin(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool=''  # Disable global pooling
        )
        
        # Get number of features from Swin-Base
        in_features = 1024
        
        # Create classifier head
        self.classifier = nn.Sequential(
            # Spatial dimension handling
            nn.LayerNorm([in_features, 7, 7]),  # Normalize features
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),  # Convert to [batch_size, features]
            
            # Dense layers (same as EfficientNet)
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the classifier head"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get features from Swin backbone
        features = self.backbone.forward_features(x)  # [B, H, W, C]
        # Permute dimensions to [B, C, H, W]
        features = features.permute(0, 3, 1, 2)
        # Apply classifier (handles normalization, pooling, and classification)
        return self.classifier(features)
    
    def get_optimizer(self):
        """Get optimizer with different learning rates for backbone and classifier"""
        # Separate backbone and classifier parameters
        backbone_params = []
        classifier_params = []
        
        # All parameters before the final classifier
        for name, param in self.named_parameters():
            if 'classifier.' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},  # Lower LR for backbone
            {'params': classifier_params, 'lr': 1e-4}  # Higher LR for classifier
        ], weight_decay=0.01)
        
        return optimizer