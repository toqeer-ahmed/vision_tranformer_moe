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
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
            
            intersection = (probs * targets_one_hot).sum(dim=(-2, -1))
            cardinality = (probs + targets_one_hot).sum(dim=(-2, -1))
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            return 1.0 - dice.mean()

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss tailored for medical segmentation with class imbalance.
    Alpha penalizes False Positives, Beta penalizes False Negatives (missed polyps).
    Gamma controls non-linear focal focusing.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        tp = (probs * targets_one_hot).sum(dim=(-2, -1))
        fp = (probs * (1.0 - targets_one_hot)).sum(dim=(-2, -1))
        fn = ((1.0 - probs) * targets_one_hot).sum(dim=(-2, -1))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1.0 - tversky) ** self.gamma
        return focal_tversky.mean()

class LoadBalancingLoss(nn.Module):
    """
    Load Balancing Loss based on Coefficient of Variation (CV) 
    to encourage balanced utilization of experts in MoTE routing.
    """
    def __init__(self):
        super(LoadBalancingLoss, self).__init__()

    def forward(self, gating_probs: torch.Tensor) -> torch.Tensor:
        # gating_probs shape: (batch_size * sequence_length, num_experts)
        # importance = sum of probabilities for each expert
        importance = gating_probs.sum(dim=0)
        # load = number of tokens routed to each expert (approx using probs > 0 or max)
        # For simplicity and differentiable routing, we use soft load (mean probs)
        mean_importance = importance.mean()
        var_importance = importance.var(unbiased=False)
        cv_squared = var_importance / (mean_importance ** 2 + 1e-6)
        
        return cv_squared


class SobelBoundaryLoss(nn.Module):
    """
    Sobel Boundary Loss for sharpening edge spatial boundaries in medical segmentation.
    """
    def __init__(self):
        super(SobelBoundaryLoss, self).__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)[:, 1:2, :, :] # Foreground channel
        targets_float = (targets == 1).unsqueeze(1).float()
        
        pred_grad_x = F.conv2d(probs, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(probs, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
        
        target_grad_x = F.conv2d(targets_float, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(targets_float, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)
        
        return F.l1_loss(pred_edges, target_edges)

class CombinedSegmentationLoss(nn.Module):
    """
    Unified Compound Loss Function supporting Cross-Entropy, Dice Loss, and Focal Tversky Loss.
    """
    def __init__(self, loss_type: str = "combined", dice_weight: float = 1.0, ce_weight: float = 1.0, balance_weight: float = 0.01):
        super(CombinedSegmentationLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.balance_weight = balance_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_tversky_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
        self.boundary_loss = SobelBoundaryLoss()
        self.load_balancing_loss = LoadBalancingLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, gating_probs: torch.Tensor = None) -> torch.Tensor:
        if self.loss_type == "focal_tversky":
            ce = self.ce_loss(logits, targets)
            ft = self.focal_tversky_loss(logits, targets)
            edge = self.boundary_loss(logits, targets)
            loss = (self.ce_weight * ce) + (self.dice_weight * ft) + (0.5 * edge)
        elif self.loss_type == "dice":
            loss = self.dice_loss(logits, targets)
        else:
            ce = self.ce_loss(logits, targets)
            dice = self.dice_loss(logits, targets)
            loss = (self.ce_weight * ce) + (self.dice_weight * dice)
            
        if gating_probs is not None:
            l_balance = self.load_balancing_loss(gating_probs)
            loss = loss + (self.balance_weight * l_balance)
            
        return loss
