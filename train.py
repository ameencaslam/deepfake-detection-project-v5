import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from PIL import Image
import logging
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class VideoFrameDataset(Dataset):
    def __init__(self, paths, labels, sequence_length=16, transform=None):
        self.paths = paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.sequence_length:
            frame_indices = list(range(total_frames))
            frame_indices.extend([total_frames-1] * (self.sequence_length - total_frames))
        else:
            frame_indices = sorted(random.sample(range(total_frames), self.sequence_length))

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            else:
                frames.append(torch.zeros_like(frames[0]) if frames else torch.zeros(3, 224, 224))
        
        cap.release()
        return torch.stack(frames)

    def load_frames_from_directory(self, frame_dir):
        frames = []
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
        
        if len(frame_files) < self.sequence_length:
            frame_indices = list(range(len(frame_files)))
            frame_indices.extend([len(frame_files)-1] * (self.sequence_length - len(frame_files)))
        else:
            frame_indices = sorted(random.sample(range(len(frame_files)), self.sequence_length))
        
        for idx in frame_indices:
            frame_path = os.path.join(frame_dir, frame_files[idx])
            try:
                frame = Image.open(frame_path).convert('RGB')
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            except Exception as e:
                logger.error(f"Error loading frame {frame_path}: {str(e)}")
                frames.append(torch.zeros_like(frames[0]) if frames else torch.zeros(3, 224, 224))
        
        return torch.stack(frames)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        
        try:
            if os.path.isfile(path) and path.endswith(('.mp4', '.avi')):
                frames = self.extract_frames(path)
            else:  # Assume it's a directory of frames
                frames = self.load_frames_from_directory(path)
            return frames, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error processing {path}: {str(e)}")
            return torch.zeros(self.sequence_length, 3, 224, 224), torch.tensor(label, dtype=torch.long)

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        
        self.efficient_net = models.efficientnet_b0(pretrained=True)
        feature_size = self.efficient_net.classifier[1].in_features
        self.efficient_net = nn.Sequential(*list(self.efficient_net.children())[:-1])
        
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, sequence_length, c, h, w = x.size()
        
        cnn_input = x.view(-1, c, h, w)
        
        cnn_features = self.efficient_net(cnn_input)
        cnn_features = cnn_features.view(batch_size, sequence_length, -1)
        
        lstm_out, _ = self.lstm(cnn_features)
        
        lstm_features = lstm_out[:, -1, :]
        
        output = self.fc(lstm_features)
        return output

class DeepfakeTrainer:
    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{total_loss/len(pbar):.3f}',
                'Acc': f'{100.*correct/total:.1f}'
            })
        
        return total_loss/len(self.train_loader), correct/total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        pbar = tqdm(self.val_loader, desc='Validating')
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = outputs.max(1)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                correct = sum(np.array(all_predictions) == np.array(all_targets))
                total = len(all_targets)
                pbar.set_postfix({
                    'Loss': f'{total_loss/len(pbar):.3f}',
                    'Acc': f'{100.*correct/total:.1f}'
                })
        
        if self.scheduler is not None:
            self.scheduler.step(total_loss/len(self.val_loader))
        
        return (total_loss/len(self.val_loader), 
                correct/total,
                np.array(all_targets),
                np.array(all_predictions),
                np.array(all_probabilities))

class MetricsLogger:
    def __init__(self, save_dir='metrics'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.epochs = []
        
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
    
    def plot_training_curves(self):
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.train_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.train_accs, label='Train Acc')
        plt.plot(self.epochs, self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_video_paths_and_labels(data_dir):
    video_paths = []
    labels = []
    
    real_dir = os.path.join(data_dir, 'real')
    if os.path.exists(real_dir):
        for video in os.listdir(real_dir):
            if video.endswith(('.mp4', '.avi')):
                video_paths.append(os.path.join(real_dir, video))
                labels.append(0)
    
    fake_dir = os.path.join(data_dir, 'fake')
    if os.path.exists(fake_dir):
        for video in os.listdir(fake_dir):
            if video.endswith(('.mp4', '.avi')):
                video_paths.append(os.path.join(fake_dir, video))
                labels.append(1)
    
    return video_paths, labels

def get_frame_dirs_and_labels(data_dir):
    frame_dirs = []
    labels = []
    
    real_faces_dir = os.path.join(data_dir, 'real_faces')
    if os.path.exists(real_faces_dir):
        for video_frames in os.listdir(real_faces_dir):
            frame_dir = os.path.join(real_faces_dir, video_frames)
            if os.path.isdir(frame_dir):
                frame_dirs.append(frame_dir)
                labels.append(0)
    
    fake_faces_dir = os.path.join(data_dir, 'fake_faces')
    if os.path.exists(fake_faces_dir):
        for video_frames in os.listdir(fake_faces_dir):
            frame_dir = os.path.join(fake_faces_dir, video_frames)
            if os.path.isdir(frame_dir):
                frame_dirs.append(frame_dir)
                labels.append(1)
    
    return frame_dirs, labels

def main():
    set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    video_data_dir = "path_to_video_dataset"  
    frame_data_dir = "path_to_frame_dataset"  
    
    video_paths, video_labels = get_video_paths_and_labels(video_data_dir)
    frame_dirs, frame_labels = get_frame_dirs_and_labels(frame_data_dir)
    
    all_paths = video_paths + frame_dirs
    all_labels = video_labels + frame_labels
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    train_dataset = VideoFrameDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = VideoFrameDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model = DeepfakeDetector().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    trainer = DeepfakeTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    metrics_dir = 'metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_logger = MetricsLogger(metrics_dir)
    
    num_epochs = 20
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = trainer.train_epoch()
        
        val_loss, val_acc, val_targets, val_predictions, val_probabilities = trainer.validate()
        
        metrics_logger.update(epoch + 1, train_loss, train_acc, val_loss, val_acc)
        
        logger.info(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(metrics_dir, 'best_model.pth'))
            
            plot_confusion_matrix(
                val_targets, 
                val_predictions,
                os.path.join(metrics_dir, 'confusion_matrix.png')
            )
            plot_roc_curve(
                val_targets,
                val_probabilities,
                os.path.join(metrics_dir, 'roc_curve.png')
            )
            plot_precision_recall_curve(
                val_targets,
                val_probabilities,
                os.path.join(metrics_dir, 'pr_curve.png')
            )
    
    metrics_logger.plot_training_curves()
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    main()
