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
    from vision_transformer_research.models.attention_unet import AttentionUNet
    from vision_transformer_research.evaluation.losses import CombinedSegmentationLoss
    from vision_transformer_research.evaluation.metrics import SegmentationMetricAccumulator
    from vision_transformer_research.datasets.medical_dataset import get_medical_dataloaders
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.seed import set_seed
    from utils.logger import setup_logger
    from utils.checkpoint import save_checkpoint, load_checkpoint
    from models.attention_unet import AttentionUNet
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
    experiment_name = model_cfg.get("name", "attention_unet")
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
    logger = setup_logger("attention_unet", log_dir=log_dir)
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
    
    # Initialize Attention U-Net
    logger.info("Initializing Attention U-Net (Oktay et al., 2018)...")
    model = AttentionUNet(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 3),
        base_channels=model_cfg.get("base_channels", 64),
        bilinear=model_cfg.get("bilinear", False)
    )
    model.to(device)
    
    # Log parameter counts
    param_counts = model.count_parameters()
    logger.info(f"Total parameters:          {param_counts['total_parameters']:,}")
    logger.info(f"Trainable parameters:      {param_counts['trainable_parameters']:,}")
    logger.info(f"Encoder parameters:        {param_counts['encoder_parameters']:,}")
    logger.info(f"Bottleneck parameters:     {param_counts['bottleneck_parameters']:,}")
    logger.info(f"Decoder parameters:        {param_counts['decoder_parameters']:,}")
    logger.info(f"Attention Gate parameters: {param_counts['attention_gate_parameters']:,}")
    logger.info(f"Head parameters:           {param_counts['head_parameters']:,}")
    
    # Loss Setup (Focal Tversky Loss)
    criterion = CombinedSegmentationLoss(loss_type="focal_tversky", dice_weight=1.0).to(device)
    
    # Optimizer (AdamW) & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs_to_run,
        eta_min=1e-6
    )
    
    # Tensorboard
    writer = None
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(log_dir=log_dir)
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard SummaryWriter: {e}")
            writer = None
    
    # Scaler for AMP
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Training state
    best_val_miou = 0.0
    patience = train_cfg.get("early_stopping_patience", 10)
    patience_counter = 0
    best_epoch = 0
    
    logger.info("Starting Attention U-Net training loop...")
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(
                    f"Epoch [{epoch}/{epochs_to_run}] Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
                
        train_loss /= len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_metric_acc = SegmentationMetricAccumulator(num_classes=model_cfg.get("num_classes", 3))
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_metric_acc.update(preds, masks)
                
        val_loss /= len(val_loader)
        val_metrics = val_metric_acc.compute()
        val_miou = val_metrics["mean_iou"]
        val_mdice = val_metrics["mean_dice"]
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        logger.info(
            f"Epoch [{epoch}/{epochs_to_run}] (LR: {current_lr:.2e}) - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou:.4f} | Val mDice: {val_mdice:.4f}"
        )
        
        if writer is not None:
            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/mIoU", val_miou, epoch)
            writer.add_scalar("Val/mDice", val_mdice, epoch)
            writer.add_scalar("LR", current_lr, epoch)
            
        # Checkpoint selection (best validation mIoU)
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            patience_counter = 0
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"--> Saved new best model checkpoint (Val mIoU: {val_miou:.4f}) to {best_ckpt_path}")
        else:
            patience_counter += 1
            logger.info(f"Early stopping counter: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best Val mIoU: {best_val_miou:.4f} (Epoch {best_epoch}).")
            break

    # Final In-domain evaluation on test set using best checkpoint
    logger.info("Evaluating best model checkpoint on held-in BUSI test split...")
    best_ckpt = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        
    model.eval()
    test_metric_acc = SegmentationMetricAccumulator(num_classes=model_cfg.get("num_classes", 3))
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_metric_acc.update(preds, masks)
            
    test_metrics = test_metric_acc.compute()
    logger.info(
        f"BUSI Test Results (Best Epoch {best_epoch}) - "
        f"mIoU: {test_metrics['mean_iou']:.4f} | mDice: {test_metrics['mean_dice']:.4f} | "
        f"Pixel Acc: {test_metrics['pixel_accuracy']:.4f}"
    )
    
    # Save test metrics JSON
    import json
    with open(os.path.join(log_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    if writer is not None:
        writer.close()
    logger.info(f"Training completed successfully for Attention U-Net (Seed {train_cfg['seed']}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention U-Net Baseline on BUSI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    train(args.config)
