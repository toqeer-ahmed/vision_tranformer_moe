import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Adjust imports to allow executing as a direct script or as a module
try:
    from vision_transformer_research.utils.seed import set_seed
    from vision_transformer_research.utils.logger import setup_logger
    from vision_transformer_research.utils.checkpoint import save_checkpoint, load_checkpoint
    from vision_transformer_research.datasets.segmentation_dataset import get_segmentation_dataloaders
    from vision_transformer_research.models.segformer import SegFormerSegmentation
    from vision_transformer_research.evaluation.metrics import compute_segmentation_metrics
    from vision_transformer_research.evaluation.visualize_predictions import plot_segmentation_curves, plot_segmentation_predictions
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.seed import set_seed
    from utils.logger import setup_logger
    from utils.checkpoint import save_checkpoint, load_checkpoint
    from datasets.segmentation_dataset import get_segmentation_dataloaders
    from models.segformer import SegFormerSegmentation
    from evaluation.metrics import compute_segmentation_metrics
    from evaluation.visualize_predictions import plot_segmentation_curves, plot_segmentation_predictions

def train(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Extract details
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    log_cfg = config["logging"]
    
    # Set seed
    set_seed(train_cfg["seed"])
    
    # Setup logger
    logger = setup_logger("segmentation", log_dir=log_cfg["log_dir"])
    logger.info(f"Loaded config from {config_path}")
    
    # Check for fast_dev_run flag
    fast_dev_run = train_cfg.get("fast_dev_run", False)
    if fast_dev_run:
        logger.warning("FAST_DEV_RUN is enabled! Training will run for 1 epoch and only a few batches.")
        epochs_to_run = 1
    else:
        epochs_to_run = train_cfg["epochs"]
    
    # Setup device
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Dataloaders
    logger.info("Initializing datasets and dataloaders...")
    if fast_dev_run:
        logger.warning("Using random mock data for fast development run to bypass slow dataset downloads.")
        mock_images = torch.randn(10, 3, model_cfg["img_size"], model_cfg["img_size"])
        mock_masks = torch.randint(0, model_cfg["num_classes"], (10, model_cfg["img_size"], model_cfg["img_size"]))
        mock_dataset = torch.utils.data.TensorDataset(mock_images, mock_masks)
        train_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=dataset_cfg["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=dataset_cfg["batch_size"], shuffle=False)
        test_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=dataset_cfg["batch_size"], shuffle=False)
    else:
        if dataset_cfg["name"].lower() == "medical-image-mask":
            try:
                from datasets.medical_dataset import get_medical_dataloaders
            except ImportError:
                from vision_transformer_research.datasets.medical_dataset import get_medical_dataloaders
            
            train_loader, val_loader = get_medical_dataloaders(
                data_dir=dataset_cfg["data_dir"],
                batch_size=dataset_cfg["batch_size"],
                img_size=model_cfg["img_size"],
                num_workers=dataset_cfg["num_workers"],
                seed=train_cfg["seed"]
            )
            test_loader = val_loader
        else:
            train_loader, val_loader, test_loader = get_segmentation_dataloaders(
                dataset_name=dataset_cfg["name"],
                data_dir=dataset_cfg["data_dir"],
                batch_size=dataset_cfg["batch_size"],
                img_size=model_cfg["img_size"],
                num_workers=dataset_cfg["num_workers"],
                seed=train_cfg["seed"]
            )
    
    # Model
    logger.info(f"Loading SegFormer model: {model_cfg['name']}...")
    model = SegFormerSegmentation(
        model_name=model_cfg["name"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"]
    )
    model.to(device)
    
    # Print model params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"])
    )
    
    # Tensorboard writer
    tb_writer = None
    if log_cfg["use_tensorboard"]:
        tb_writer = SummaryWriter(log_dir=log_cfg["log_dir"])
        
    # Epoch records
    train_losses, val_losses = [], []
    val_dices, val_ious = [], []
    
    best_val_iou = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping_patience"]
    
    logger.info("Starting training loop...")
    for epoch in range(1, epochs_to_run + 1):
        # Training epoch
        model.train()
        running_loss = 0.0
        total_pixels = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            if fast_dev_run and batch_idx >= 2:
                break
                
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(images) # shape: (B, C, H, W)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            total_pixels += images.size(0)
            
            if (batch_idx + 1) % 10 == 0 or fast_dev_run:
                logger.info(f"Epoch [{epoch}/{epochs_to_run}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / total_pixels
        train_losses.append(epoch_train_loss)
        
        # Validation epoch
        model.eval()
        running_val_loss = 0.0
        val_pixels = 0
        
        all_preds = []
        all_targets = []
        
        # Save sample images for visualization
        sample_images = None
        sample_masks = None
        sample_preds = None
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                if fast_dev_run and batch_idx >= 2:
                    break
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                
                running_val_loss += loss.item() * images.size(0)
                val_pixels += images.size(0)
                
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                
                if batch_idx == 0:
                    sample_images = images.cpu()
                    sample_masks = targets.cpu()
                    sample_preds = preds.cpu()
                    
        epoch_val_loss = running_val_loss / val_pixels
        val_losses.append(epoch_val_loss)
        
        # Concatenate and compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_segmentation_metrics(all_preds, all_targets, num_classes=model_cfg["num_classes"])
        val_ious.append(metrics["mean_iou"])
        val_dices.append(metrics["mean_dice"])
        
        logger.info(
            f"Epoch [{epoch}/{epochs_to_run}] - "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val mIoU: {metrics['mean_iou']:.4f} | "
            f"Val mDice: {metrics['mean_dice']:.4f} | "
            f"Pixel Acc: {metrics['pixel_accuracy']:.4f}"
        )
        
        # Tensorboard log
        if tb_writer:
            tb_writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
            tb_writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
            tb_writer.add_scalar("Metrics/mIoU", metrics["mean_iou"], epoch)
            tb_writer.add_scalar("Metrics/mDice", metrics["mean_dice"], epoch)
            tb_writer.add_scalar("Metrics/PixelAccuracy", metrics["pixel_accuracy"], epoch)
            
        # Checkpoint save
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou": best_val_iou,
        }
        save_checkpoint(state, log_cfg["checkpoint_dir"], filename="last_checkpoint.pth")
        
        if metrics["mean_iou"] > best_val_iou or fast_dev_run:
            best_val_iou = metrics["mean_iou"]
            patience_counter = 0
            save_checkpoint(state, log_cfg["checkpoint_dir"], filename="best_model.pth")
            logger.info(f"New best validation mIoU: {best_val_iou:.4f}. Saved best model checkpoint.")
            
            # Save sample predictions on best model
            if sample_images is not None:
                plot_path = os.path.join(log_cfg["plot_dir"], f"val_predictions_epoch_{epoch}.png")
                plot_segmentation_predictions(sample_images, sample_masks, sample_preds, plot_path)
                logger.info(f"Saved validation predictions visualization to {plot_path}")
        else:
            patience_counter += 1
            logger.info(f"Validation mIoU did not improve. Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered. Training stopped.")
                break
                
    # Plot curves
    plot_segmentation_curves(train_losses, val_losses, val_ious, val_dices, log_cfg["plot_dir"])
    
    # Testing
    logger.info("Loading best model for testing...")
    best_path = os.path.join(log_cfg["checkpoint_dir"], "best_model.pth")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, device=device)
        
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if fast_dev_run and batch_idx >= 2:
                break
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            test_preds.append(preds)
            test_targets.append(targets)
            
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    test_metrics = compute_segmentation_metrics(test_preds, test_targets, num_classes=model_cfg["num_classes"])
    logger.info(
        f"Test Set Metrics - "
        f"mIoU: {test_metrics['mean_iou']:.4f} | "
        f"mDice: {test_metrics['mean_dice']:.4f} | "
        f"Pixel Acc: {test_metrics['pixel_accuracy']:.4f} | "
        f"Precision: {test_metrics['mean_precision']:.4f} | "
        f"Recall: {test_metrics['mean_recall']:.4f}"
    )
    
    if tb_writer:
        tb_writer.close()
        
    logger.info("Training process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer Semantic Segmentation")
    parser.add_argument("--config", type=str, default="configs/segmentation.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)
