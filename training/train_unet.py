import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, Exception):
    SummaryWriter = None

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Adjust imports to allow executing as a direct script or as a module
try:
    from vision_transformer_research.utils.seed import set_seed
    from vision_transformer_research.utils.logger import setup_logger
    from vision_transformer_research.utils.checkpoint import save_checkpoint, load_checkpoint
    from vision_transformer_research.models.unet import UNet
    from vision_transformer_research.evaluation.losses import CombinedSegmentationLoss
    from vision_transformer_research.evaluation.metrics import SegmentationMetricAccumulator
    from vision_transformer_research.datasets.medical_dataset import get_medical_dataloaders
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.seed import set_seed
    from utils.logger import setup_logger
    from utils.checkpoint import save_checkpoint, load_checkpoint
    from models.unet import UNet
    from evaluation.losses import CombinedSegmentationLoss
    from evaluation.metrics import SegmentationMetricAccumulator
    from datasets.medical_dataset import get_medical_dataloaders

def train(config_path: str):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    model_cfg = config["model"]
    train_cfg = config["training"]
    
    # Generate unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = model_cfg.get("name", "unet_standard")
    experiment_dir = os.path.join("outputs", f"{experiment_name}_{timestamp}")
    log_dir = os.path.join(experiment_dir, "logs")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    plot_dir = os.path.join(experiment_dir, "plots")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set seed across all RNGs
    set_seed(train_cfg["seed"])
    
    # Setup logger
    logger = setup_logger("unet_standard", log_dir=log_dir)
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Experiment Directory: {experiment_dir}")
    
    epochs_to_run = train_cfg["num_epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize Dataloaders (BUSI 80/10/10 split)
    logger.info("Initializing BUSI datasets and dataloaders...")
    workers = 2 if os.name != "nt" else 0
    train_loader, val_loader, test_loader = get_medical_dataloaders(
        data_dir="data/medical_dataset",
        batch_size=train_cfg["batch_size"],
        img_size=train_cfg["img_size"],
        num_workers=workers,
        seed=train_cfg["seed"]
    )
    
    # Initialize Standard U-Net
    logger.info("Initializing Canonical 2D U-Net (Ronneberger et al., 2015)...")
    model = UNet(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 3),
        base_channels=model_cfg.get("base_channels", 64),
        bilinear=model_cfg.get("bilinear", False)
    )
    model.to(device)
    
    # Log parameter counts
    param_counts = model.count_parameters()
    logger.info(f"Total parameters:      {param_counts['total_parameters']:,}")
    logger.info(f"Trainable parameters:  {param_counts['trainable_parameters']:,}")
    logger.info(f"Encoder parameters:    {param_counts['encoder_parameters']:,}")
    logger.info(f"Bottleneck parameters: {param_counts['bottleneck_parameters']:,}")
    logger.info(f"Decoder parameters:    {param_counts['decoder_parameters']:,}")
    logger.info(f"Head parameters:       {param_counts['head_parameters']:,}")
    
    # Loss Setup (Focal Tversky Loss)
    criterion = CombinedSegmentationLoss(loss_type="focal_tversky", dice_weight=1.0).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"])
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_to_run, eta_min=1e-6
    )
    
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    tb_writer = SummaryWriter(log_dir=log_dir) if SummaryWriter is not None else None
    
    # Training records
    train_losses, val_losses = [], []
    val_dices, val_ious = [], []
    
    best_val_iou = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping_patience"]
    
    logger.info("Starting training loop...")
    start_time = datetime.now()
    
    for epoch in range(1, epochs_to_run + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        total_pixels = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(images)
                loss = criterion(logits, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            b_size = images.size(0)
            running_loss += loss.item() * b_size
            total_pixels += b_size
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch}/{epochs_to_run}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / total_pixels
        train_losses.append(epoch_train_loss)
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_pixels = 0
        val_accumulator = SegmentationMetricAccumulator(num_classes=model_cfg["num_classes"])
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits = model(images)
                    loss = criterion(logits, targets)
                    
                b_size = images.size(0)
                running_val_loss += loss.item() * b_size
                val_pixels += b_size
                
                preds = logits.argmax(dim=1)
                val_accumulator.update(preds, targets)
                
        epoch_val_loss = running_val_loss / val_pixels
        val_metrics = val_accumulator.compute()
        val_iou = val_metrics["mean_iou"]
        val_dice = val_metrics["mean_dice"]
        
        val_losses.append(epoch_val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)
        
        logger.info(
            f"Epoch [{epoch}/{epochs_to_run}] (LR: {current_lr:.2e}) - "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val mIoU: {val_iou:.4f} | "
            f"Val mDice: {val_dice:.4f}"
        )
        
        if tb_writer:
            tb_writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
            tb_writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
            tb_writer.add_scalar("Metrics/Val_mIoU", val_iou, epoch)
            tb_writer.add_scalar("Metrics/Val_mDice", val_dice, epoch)
            tb_writer.add_scalar("Training/LR", current_lr, epoch)
            
        # Checkpoint Management
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            # Save compact state dict
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            logger.info(f"Saved new best model with Val mIoU: {best_val_iou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience: {patience})")
                break
                
    total_duration = datetime.now() - start_time
    logger.info(f"Training completed in {total_duration}")
    
    # 3. In-Domain Testing
    logger.info("Loading best model for in-domain BUSI testing...")
    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        
    model.eval()
    test_accumulator = SegmentationMetricAccumulator(num_classes=model_cfg["num_classes"])
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(images)
            preds = logits.argmax(dim=1)
            test_accumulator.update(preds, targets)
            
    test_metrics = test_accumulator.compute()
    logger.info(
        f"In-Domain BUSI Test Metrics - "
        f"mIoU: {test_metrics['mean_iou']:.4f} | "
        f"mDice: {test_metrics['mean_dice']:.4f} | "
        f"Pixel Acc: {test_metrics['pixel_accuracy']:.4f} | "
        f"Precision: {test_metrics['mean_precision']:.4f} | "
        f"Recall: {test_metrics['mean_recall']:.4f}"
    )
    
    import json
    with open(os.path.join(log_dir, "test_metrics.json"), "w") as f:
        clean = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in test_metrics.items()}
        json.dump(clean, f, indent=4)
        
    if tb_writer:
        tb_writer.close()
        
    logger.info("U-Net training and in-domain evaluation process successfully finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Canonical 2D U-Net Semantic Segmentation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    train(args.config)
