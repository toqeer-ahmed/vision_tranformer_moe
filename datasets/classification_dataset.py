import os
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split

class CIFARDataset(Dataset):
    """
    Standard wrapper for CIFAR test/val raw datasets to use Albumentations transforms.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_np = np.array(img)
        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        return img_tensor, label

class SubsetWrapper(Dataset):
    """
    Wrapper for dataset Subsets to apply split-specific Albumentations transforms.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img_np = np.array(img)
        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        return img_tensor, label

def get_classification_transforms(img_size: int = 224):
    """
    Returns albumentations transforms for classification.
    We resize small CIFAR images (32x32) to 224x224 for standard ViT models.
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def get_classification_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 2,
    seed: int = 42
):
    """
    Helper function to prepare CIFAR-10/100 datasets and return training,
    validation, and test dataloaders.
    
    Args:
        dataset_name (str): 'cifar10' or 'cifar100'
        data_dir (str): directory to download and save the data
        batch_size (int): batch size for loaders
        img_size (int): target size to resize images
        num_workers (int): number of data loader workers
        seed (int): seed for train/val split generator
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform, val_transform = get_classification_transforms(img_size)
    
    # Select correct dataset class
    if dataset_name.lower() == "cifar10":
        dataset_cls = CIFAR10
    elif dataset_name.lower() == "cifar100":
        dataset_cls = CIFAR100
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Choose 'cifar10' or 'cifar100'.")
        
    full_train_dataset = dataset_cls(root=data_dir, train=True, download=True)
    test_raw_dataset = dataset_cls(root=data_dir, train=False, download=True)
    
    # Partition full training set into 90% training and 10% validation
    train_len = int(0.9 * len(full_train_dataset))
    val_len = len(full_train_dataset) - train_len
    
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_len, val_len], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_dataset = SubsetWrapper(train_subset, train_transform)
    val_dataset = SubsetWrapper(val_subset, val_transform)
    test_dataset = CIFARDataset(test_raw_dataset, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
