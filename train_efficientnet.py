# train_efficientnet.py
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from data_handler import get_dataloaders
import logging
import os
from tqdm import tqdm
from torchviz import make_dot
import torchinfo
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class DeepfakeTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = model.get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1
        )
        
        # Add these to store learning curves data
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        predictions = []
        targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = ((np.array(predictions) > 0.5) == np.array(targets)).mean()
        
        return epoch_loss, epoch_acc, predictions, targets
    
    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                predictions.extend(torch.sigmoid(outputs).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        avg_loss = running_loss / len(loader)
        accuracy = ((np.array(predictions) > 0.5) == np.array(targets)).mean()
        
        return avg_loss, accuracy, predictions, targets
    
    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        mlflow.log_metrics({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }, step=epoch)
    
    def log_plots(self, y_true, y_pred, phase='train'):
        # Confusion Matrix
        cm = confusion_matrix(y_true, np.array(y_pred) > 0.5)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'{phase} Confusion Matrix')
        mlflow.log_figure(plt.gcf(), f'{phase}_confusion_matrix.png')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{phase} ROC Curve')
        plt.legend()
        mlflow.log_figure(plt.gcf(), f'{phase}_roc_curve.png')
        plt.close()
    
    def plot_learning_curves(self):
        """Plot and save learning curves."""
        epochs = range(len(self.train_losses))
        
        # Create figure with secondary y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'b--', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'r-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r--', label='Validation Accuracy')
        ax2.set_ylabel('Accuracy', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.title('Training and Validation Metrics')
        plt.tight_layout()
        
        # Log the figure to MLflow
        mlflow.log_figure(fig, "learning_curves.png")
        plt.close()
    
    def log_model_architecture(self):
        """Log model architecture visualization and summary."""
        # Create directory for artifacts if it doesn't exist
        Path("model_artifacts").mkdir(exist_ok=True)
        
        try:
            # Generate model summary using torchinfo with batch_size=2 (to avoid BatchNorm error)
            model_stats = torchinfo.summary(
                self.model, 
                input_size=(2, 3, 224, 224),  # Changed batch size to 2
                verbose=0,
                device=self.device
            )
            
            # Save model summary to file
            with open("model_artifacts/model_summary.txt", "w") as f:
                f.write(str(model_stats))
            
            # Generate model visualization using torchviz
            dummy_input = torch.zeros(2, 3, 224, 224, device=self.device)  # Changed batch size to 2
            with torch.no_grad():
                output = self.model(dummy_input)
                dot = make_dot(output, params=dict(self.model.named_parameters()))
                dot.render("model_artifacts/model_architecture", format="png")
            
            # Log artifacts to MLflow
            mlflow.log_artifact("model_artifacts/model_summary.txt")
            mlflow.log_artifact("model_artifacts/model_architecture.png")
        
        except Exception as e:
            logger.warning(f"Failed to log model architecture: {str(e)}")
            # Continue training even if visualization fails
            pass
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        
        # Log model architecture at the start of training
        self.log_model_architecture()
        
        for epoch in range(num_epochs):
            train_loss, train_acc, train_preds, train_targets = self.train_epoch(epoch)
            val_loss, val_acc, val_preds, val_targets = self.validate(self.val_loader)
            
            # Store metrics for learning curves
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log metrics
            self.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
            
            # Log plots every few epochs
            if epoch % 5 == 0:
                self.log_plots(train_targets, train_preds, 'train')
                self.log_plots(val_targets, val_preds, 'validation')
            
            # Model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Save GPU version (existing MLflow logging)
                mlflow.pytorch.log_model(self.model, "best_model")
                
                # Save CPU-compatible version directly
                torch.save(self.model.state_dict(), "best_model_cpu.pth", map_location='cpu')
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
                       f'Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, '
                       f'Val Acc={val_acc:.4f}')
        
        # Plot learning curves at the end of training
        self.plot_learning_curves()
    
    def test(self):
        test_loss, test_acc, test_preds, test_targets = self.validate(self.test_loader)
        self.log_plots(test_targets, test_preds, 'test')
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_accuracy': test_acc
        })
        return test_loss, test_acc

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # MLflow setup
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment('deepfake_efficientnet')
    
    # Configuration
    DATA_DIR = '/kaggle/input/2body-images-10k-split'  # Adjust as needed
    IMAGE_SIZE = 300  # EfficientNet-B3 optimal size
    BATCH_SIZE = 32  # Reduced batch size for larger model
    NUM_EPOCHS = 15
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_DIR, IMAGE_SIZE, BATCH_SIZE
    )
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'model_type': 'efficientnet-b3',
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'optimizer': 'AdamW',
            'backbone_lr': 1e-5,
            'classifier_lr': 5e-5,
            'weight_decay': 0.01
        })
        
        # Create and train model
        model = DeepfakeEfficientNet()
        trainer = DeepfakeTrainer(model, train_loader, val_loader, test_loader, DEVICE)
        
        # Train
        trainer.train(NUM_EPOCHS)
        
        # Test
        test_loss, test_acc = trainer.test()
        logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main()