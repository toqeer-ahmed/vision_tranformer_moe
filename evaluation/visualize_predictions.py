import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    save_path: str,
    title: str = "Confusion Matrix",
    cmap=plt.cm.Blues
):
    """
    Plots and saves the confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix computed using sklearn.
        classes (list): List of class names.
        save_path (str): Filepath to save the plot.
        title (str): Title of the plot.
        cmap: Colormap for plot.
    """
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Format values in cell
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_classification_curves(train_losses: list, val_losses: list, train_accs: list, val_accs: list, save_dir: str):
    """
    Plots and saves the training and validation loss & accuracy curves.
    
    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        train_accs (list): Training accuracy per epoch.
        val_accs (list): Validation accuracy per epoch.
        save_dir (str): Directory path to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.close()
    
    # Accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, 'b-o', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curves.png"))
    plt.close()

def plot_segmentation_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    save_path: str,
    num_samples: int = 4
):
    """
    Plots raw images, ground-truth masks, and predicted masks side-by-side.
    
    Args:
        images (torch.Tensor): Image tensor of shape (N, C, H, W).
        masks (torch.Tensor): Ground-truth mask tensor of shape (N, H, W).
        preds (torch.Tensor): Predicted mask tensor of shape (N, H, W).
        save_path (str): Filepath to save the plot.
        num_samples (int): Number of sample predictions to visualize.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(len(images), num_samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(n):
        # Denormalize image (assuming standard ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = images[i].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        gt = masks[i].cpu().numpy()
        pred = preds[i].cpu().numpy()
        
        # Plot Input Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")
        
        # Plot Ground Truth Mask
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")
        
        # Plot Predicted Mask
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_segmentation_curves(
    train_losses: list,
    val_losses: list,
    val_ious: list,
    val_dices: list,
    save_dir: str
):
    """
    Plots and saves segmentation training/val loss and IoU/Dice curves.
    
    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        val_ious (list): Validation IoU per epoch.
        val_dices (list): Validation Dice score per epoch.
        save_dir (str): Directory path to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.close()
    
    # Metrics curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_ious, 'g-s', label='Validation IoU')
    plt.plot(epochs, val_dices, 'm-^', label='Validation Dice')
    plt.title('Validation IoU and Dice Scores')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "segmentation_metrics.png"))
    plt.close()
