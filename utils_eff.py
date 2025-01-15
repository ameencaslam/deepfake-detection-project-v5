import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class DeepfakeEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained model
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3', num_classes=0)
        
        # Get the number of features from the backbone
        in_features = 1536  # B3 features
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # First block: in_features -> 1024
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=0.5),
            
            # Second block: 1024 -> 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            
            # Third block: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            
            # Output layer: 256 -> 1
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
        # Extract features
        features = self.backbone.extract_features(x)
        # Apply classifier
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