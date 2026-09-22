import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_classification_metrics(all_preds: np.ndarray, all_targets: np.ndarray, average: str = "macro"):
    """
    Computes classification metrics using sklearn.
    
    Args:
        all_preds (np.ndarray): Predictions (integers)
        all_targets (np.ndarray): Target labels (integers)
        average (str): Averaging method for multi-class metrics ('macro', 'micro', 'weighted')
        
    Returns:
        dict: Dict containing accuracy, precision, recall, f1_score, and confusion_matrix.
    """
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=average, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm
    }

def compute_segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100
):
    """
    Computes Dice Score, IoU, Pixel Accuracy, Precision, and Recall for segmentation.
    Supports binary and multi-class.
    
    Args:
        preds (torch.Tensor): Predicted class labels, shape (N, H, W) or (N*H*W)
        targets (torch.Tensor): Ground truth labels, shape (N, H, W) or (N*H*W)
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in metric calculations
        
    Returns:
        dict: Dict containing metrics.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Mask out ignored index
    if ignore_index is not None:
        valid_mask = (targets != ignore_index)
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
    total = len(targets)
    if total == 0:
        return {
            "pixel_accuracy": 0.0,
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "class_iou": [0.0] * num_classes,
            "class_dice": [0.0] * num_classes
        }
        
    # Pixel Accuracy
    correct = (preds == targets).sum().item()
    pixel_acc = correct / total
    
    # Class-wise metrics
    ious = []
    dices = []
    precisions = []
    recalls = []
    
    tps = []
    fps = []
    fns = []
    
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        
        tp = (pred_c & target_c).sum().item()
        fp = (pred_c & ~target_c).sum().item()
        fn = (~pred_c & target_c).sum().item()
        
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        
        # IoU (Jaccard Index)
        union = tp + fp + fn
        if union > 0:
            iou = tp / union
        else:
            iou = 1.0 if (target_c.sum().item() == 0 and pred_c.sum().item() == 0) else 0.0
        ious.append(iou)
        
        # Dice Score (F1 Score)
        denominator = 2 * tp + fp + fn
        if denominator > 0:
            dice = (2 * tp) / denominator
        else:
            dice = 1.0 if (target_c.sum().item() == 0 and pred_c.sum().item() == 0) else 0.0
        dices.append(dice)
        
        # Precision
        precision_denom = tp + fp
        prec = tp / precision_denom if precision_denom > 0 else 0.0
        precisions.append(prec)
        
        # Recall
        recall_denom = tp + fn
        rec = tp / recall_denom if recall_denom > 0 else 0.0
        recalls.append(rec)
        
    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "class_iou": [float(i) for i in ious],
        "class_dice": [float(d) for d in dices],
        "class_precision": [float(p) for p in precisions],
        "class_recall": [float(r) for r in recalls],
        "class_tp": [int(x) for x in tps],
        "class_fp": [int(x) for x in fps],
        "class_fn": [int(x) for x in fns]
    }

class SegmentationMetricAccumulator:
    """
    Memory-efficient accumulator for segmentation metrics.
    Avoids retaining all predictions in RAM by streaming batch statistics.
    """
    def __init__(self, num_classes=2, ignore_index=-100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        self.tps = np.zeros(num_classes, dtype=np.int64)
        self.fps = np.zeros(num_classes, dtype=np.int64)
        self.fns = np.zeros(num_classes, dtype=np.int64)
        
        self.total_pixels = 0
        self.correct_pixels = 0
        
    def reset(self):
        self.tps.fill(0)
        self.fps.fill(0)
        self.fns.fill(0)
        self.total_pixels = 0
        self.correct_pixels = 0
        
    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            preds = preds[valid_mask]
            targets = targets[valid_mask]
            
        self.total_pixels += len(targets)
        if len(targets) == 0:
            return
            
        self.correct_pixels += (preds == targets).sum().item()
        
        for c in range(self.num_classes):
            pred_c = (preds == c)
            target_c = (targets == c)
            
            self.tps[c] += (pred_c & target_c).sum().item()
            self.fps[c] += (pred_c & ~target_c).sum().item()
            self.fns[c] += (~pred_c & target_c).sum().item()
            
    def compute(self):
        if self.total_pixels == 0:
            return {
                "pixel_accuracy": 0.0,
                "mean_iou": 0.0,
                "mean_dice": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "class_iou": [0.0] * self.num_classes,
                "class_dice": [0.0] * self.num_classes,
                "class_precision": [0.0] * self.num_classes,
                "class_recall": [0.0] * self.num_classes,
                "class_tp": [0] * self.num_classes,
                "class_fp": [0] * self.num_classes,
                "class_fn": [0] * self.num_classes
            }
            
        pixel_acc = self.correct_pixels / self.total_pixels
        
        ious = []
        dices = []
        precisions = []
        recalls = []
        
        for c in range(self.num_classes):
            tp = self.tps[c]
            fp = self.fps[c]
            fn = self.fns[c]
            
            # IoU
            union = tp + fp + fn
            if union > 0:
                iou = tp / union
            else:
                # If target has no instances of this class and model predicts none
                iou = 1.0 if (fn == 0 and tp == 0 and fp == 0) else 0.0
            ious.append(iou)
            
            # Dice
            denominator = 2 * tp + fp + fn
            if denominator > 0:
                dice = (2 * tp) / denominator
            else:
                dice = 1.0 if (fn == 0 and tp == 0 and fp == 0) else 0.0
            dices.append(dice)
            
            # Precision
            precision_denom = tp + fp
            prec = tp / precision_denom if precision_denom > 0 else 0.0
            precisions.append(prec)
            
            # Recall
            recall_denom = tp + fn
            rec = tp / recall_denom if recall_denom > 0 else 0.0
            recalls.append(rec)
            
        return {
            "pixel_accuracy": pixel_acc,
            "mean_iou": float(np.mean(ious)),
            "mean_dice": float(np.mean(dices)),
            "mean_precision": float(np.mean(precisions)),
            "mean_recall": float(np.mean(recalls)),
            "class_iou": [float(i) for i in ious],
            "class_dice": [float(d) for d in dices],
            "class_precision": [float(p) for p in precisions],
            "class_recall": [float(r) for r in recalls],
            "class_tp": [int(x) for x in self.tps],
            "class_fp": [int(x) for x in self.fps],
            "class_fn": [int(x) for x in self.fns]
        }
