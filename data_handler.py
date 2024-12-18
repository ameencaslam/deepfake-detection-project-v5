# data_handler.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os
import mlflow

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get paths
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        
        # Get all images (no shuffling needed as per requirement)
        self.real_images = [os.path.join('real', img) for img in os.listdir(self.real_dir)]
        self.fake_images = [os.path.join('fake', img) for img in os.listdir(self.fake_dir)]
        
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(image_size):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_dataloaders(data_dir, image_size, batch_size=32):
    transform = get_transforms(image_size)
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    # test_ratio = 0.15 (remaining)
    
    dataset = DeepfakeDataset(data_dir, transform=transform)
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    return train_loader, val_loader, test_loader