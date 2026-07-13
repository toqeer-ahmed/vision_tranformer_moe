import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalImageMaskDataset(Dataset):
    """
    Generic medical segmentation dataset that matches input images with their corresponding mask targets.
    Expects data_dir to contain:
      - data_dir/images/
      - data_dir/masks/
    """
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            raise FileNotFoundError(
                f"Data directory must contain 'images/' and 'masks/' folders. "
                f"Checked path: {data_dir}"
            )
            
        # Scan for images
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.images_dir, ext)))
            
        # Pair images with masks based on base filename stems
        self.pairs = []
        for img_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Search masks directory for a mask file that starts with this stem
            mask_pattern = os.path.join(self.masks_dir, f"{base_name}*")
            matching_masks = glob.glob(mask_pattern)
            
            # Select the first match with a valid image extension
            valid_masks = [m for m in matching_masks if m.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if len(valid_masks) > 0:
                self.pairs.append((img_path, valid_masks[0]))
                
        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No matching image-mask pairs found in {data_dir}. "
                f"Ensure filenames in 'images/' match stems in 'masks/'."
            )
            
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # Load files
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Binarize masks: pixels > 127 are mapped to 1 (object), otherwise 0 (background)
        binary_mask = np.zeros_like(mask_np, dtype=np.float32)
        binary_mask[mask_np > 127] = 1.0
        
        if self.transform:
            augmented = self.transform(image=image_np, mask=binary_mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(binary_mask).long()
            
        return image_tensor, mask_tensor

class SubsetMedicalWrapper(Dataset):
    """
    Wrapper to allow correct training/validation splits with split-specific transforms.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        parent_dataset = self.subset.dataset
        parent_idx = self.subset.indices[idx]
        
        img_path, mask_path = parent_dataset.pairs[parent_idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        binary_mask = np.zeros_like(mask_np, dtype=np.float32)
        binary_mask[mask_np > 127] = 1.0
        
        if self.transform:
            augmented = self.transform(image=image_np, mask=binary_mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(binary_mask).long()
            
        return image_tensor, mask_tensor

def get_medical_transforms(img_size: int = 256):
    """
    Returns data augmentation transforms tailored for medical image scans.
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),           # Standard for tissue scan data (orientation invariant)
        A.RandomRotate90(p=0.5),          # Multi-angle tissue alignment rotation
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def get_medical_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    img_size: int = 256,
    num_workers: int = 2,
    seed: int = 42
):
    """
    Splits the medical dataset into 90% train and 10% validation, and creates dataloaders.
    """
    train_transform, val_transform = get_medical_transforms(img_size)
    
    full_dataset = MedicalImageMaskDataset(data_dir, transform=None)
    
    # Split datasets
    train_len = int(0.9 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    
    train_subset, val_subset = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed)
    )
    
    train_dataset = SubsetMedicalWrapper(train_subset, train_transform)
    val_dataset = SubsetMedicalWrapper(val_subset, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
