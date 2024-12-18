# data_handler.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os
import mlflow
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to the dataset root (/kaggle/input/2body-images-10k-split)
            split (str): One of 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Read split files
        split_file = os.path.join(root_dir, f'{split}_split.txt')
        with open(split_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        # Get labels from paths (paths are in format 'fake/image.jpg' or 'real/image.jpg')
        self.labels = [1 if path.startswith('fake/') else 0 for path in self.image_paths]
        
        # Verify all images exist
        for img_path in self.image_paths:
            full_path = os.path.join(self.root_dir, self.split, img_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Image not found: {full_path}")
        
        logger.info(f"Loaded {split} dataset with {len(self.image_paths)} images")
        logger.info(f"Fake images: {sum(self.labels)}")
        logger.info(f"Real images: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Construct full path: root_dir/split/fake_or_real/image.jpg
        img_path = os.path.join(self.root_dir, self.split, self.image_paths[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(image_size):
    """
    Get transforms for training and validation/testing
    Args:
        image_size (int): Size to resize images to
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Training transforms with augmentation
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Testing transforms without augmentation
    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders(data_dir, image_size, batch_size=32):
    """
    Create dataloaders for training, validation, and testing
    Args:
        data_dir (str): Path to dataset root
        image_size (int): Size to resize images to
        batch_size (int): Batch size for dataloaders
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    try:
        # Get transforms
        train_transform, val_transform = get_transforms(image_size)
        
        # Create datasets
        train_dataset = DeepfakeDataset(
            data_dir, 
            split='train', 
            transform=train_transform
        )
        
        val_dataset = DeepfakeDataset(
            data_dir, 
            split='val', 
            transform=val_transform
        )
        
        test_dataset = DeepfakeDataset(
            data_dir, 
            split='test', 
            transform=val_transform
        )
        
        # Print dataset sizes
        logger.info("Dataset sizes:")
        logger.info(f"Train: {len(train_dataset)}")
        logger.info(f"Validation: {len(val_dataset)}")
        logger.info(f"Test: {len(test_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        raise

def verify_dataset(data_dir):
    """
    Verify dataset structure and print statistics
    Args:
        data_dir (str): Path to dataset root
    """
    try:
        logger.info("Verifying dataset structure...")
        
        for split in ['train', 'val', 'test']:
            # Check split file exists
            split_file = os.path.join(data_dir, f'{split}_split.txt')
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            
            # Read and verify images
            with open(split_file, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
            
            fake_count = sum(1 for path in image_paths if path.startswith('fake/'))
            real_count = sum(1 for path in image_paths if path.startswith('real/'))
            
            logger.info(f"\n{split.upper()} split:")
            logger.info(f"Fake images: {fake_count}")
            logger.info(f"Real images: {real_count}")
            logger.info(f"Total: {fake_count + real_count}")
            
            # Verify all images exist
            for img_path in image_paths:
                full_path = os.path.join(data_dir, split, img_path)
                if not os.path.exists(full_path):
                    logger.warning(f"Image not found: {full_path}")
        
        logger.info("\nDataset verification completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dataset verification: {e}")
        raise

if __name__ == '__main__':
    # Example usage
    data_dir = '/kaggle/input/2body-images-10k-split'
    verify_dataset(data_dir)