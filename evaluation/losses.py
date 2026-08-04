import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for binary and multi-class semantic segmentation.
    Computes overlap score directly: Loss = 1 - DiceCoefficient
    """
    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W) raw unnormalized network outputs
        targets: (B, H, W) ground truth target class indices
        """
        num_classes = logits.shape[1]
        
        if num_classes == 1:
            probs = torch.sigmoid(logits).squeeze(1)
            targets = targets.float()
            
            intersection = (probs * targets).sum(dim=(-2, -1))
            cardinality = (probs + targets).sum(dim=(-2, -1))
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            return 1.0 - dice.mean()
        else:
            probs = F.softmax(logits, dim=1)
            
            # One-hot encode targets: (B, H, W) -> (B, C, H, W)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
            
            intersection = (probs * targets_one_hot).sum(dim=(-2, -1))
            cardinality = (probs + targets_one_hot).sum(dim=(-2, -1))
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            
            # Average across classes and batch
            return 1.0 - dice.mean()

class CombinedSegmentationLoss(nn.Module):
    """
    Compound Loss Function combining Cross-Entropy Loss and Dice Loss.
    Loss = CE_Loss + dice_weight * Dice_Loss
    """
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super(CombinedSegmentationLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total_loss = (self.ce_weight * ce) + (self.dice_weight * dice)
        return total_loss
