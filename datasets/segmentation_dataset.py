import os
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split

class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset wrapper for semantic segmentation.
    """
    def __init__(self, root: str, split: str = "trainval", transform=None, download: bool = True):
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        
        # Convert PIL to NumPy
        image = np.array(image.convert("RGB"))
        target = np.array(target) # Values are 1 (foreground), 2 (background), 3 (border)
        
        # Map target to binary: 1 (Pet) -> 1, 2 (Background) -> 0, 3 (Border) -> 0
        mask = np.zeros_like(target, dtype=np.float32)
        mask[target == 1] = 1.0  # Pet foreground is 1
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long() # Convert mask to long tensor for CrossEntropy
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
        return image, mask

class SubsetSegmentationWrapper(Dataset):
    """
    Wrapper for segmentation dataset subsets to allow correct transforms on splits.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        # Retrieve raw image and target mask from parent dataset
        image, target = self.subset.dataset[self.subset.indices[idx]]
        
        # Convert PIL to NumPy
        image = np.array(image.convert("RGB"))
        target = np.array(target)
        
        # Map target to binary (pet vs background)
        mask = np.zeros_like(target, dtype=np.float32)
        mask[target == 1] = 1.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
        return image, mask

def get_segmentation_transforms(img_size: int = 224):
    """
    Returns albumentations transforms for semantic segmentation.
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def get_segmentation_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 2,
    seed: int = 42
):
    """
    Prepares segmentation dataloaders for the oxford-iiit-pet dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Root directory of the dataset
        batch_size (int): Size of minibatch
        img_size (int): Image size to resize to
        num_workers (int): Number of dataloader workers
        seed (int): Random seed for splitting dataset
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if dataset_name.lower() != "oxford-iiit-pet":
        raise ValueError(f"Unsupported dataset: {dataset_name}. Currently only 'oxford-iiit-pet' is supported.")
        
    train_transform, val_transform = get_segmentation_transforms(img_size)
    
    # Load dataset
    full_train_dataset = OxfordPetDataset(data_dir, split="trainval", transform=None, download=True)
    
    # Split into 90% train, 10% validation
    train_len = int(0.9 * len(full_train_dataset))
    val_len = len(full_train_dataset) - train_len
    
    train_subset, val_subset = random_split(
        full_train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed)
    )
    
    train_dataset = SubsetSegmentationWrapper(train_subset, train_transform)
    val_dataset = SubsetSegmentationWrapper(val_subset, val_transform)
    test_dataset = OxfordPetDataset(data_dir, split="test", transform=val_transform, download=True)
    
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
