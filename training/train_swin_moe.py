import os
import argparse
import yaml
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from datetime import datetime

# Adjust imports to allow executing as a direct script or as a module
try:
    from vision_transformer_research.utils.seed import set_seed
    from vision_transformer_research.utils.logger import setup_logger
    from vision_transformer_research.utils.checkpoint import save_checkpoint, load_checkpoint
    from vision_transformer_research.models.swin import SwinSegmentation, replace_swin_ffn_with_moe
    from vision_transformer_research.evaluation.metrics import compute_segmentation_metrics
    from vision_transformer_research.evaluation.visualize_predictions import plot_segmentation_curves, plot_segmentation_predictions
    from vision_transformer_research.evaluation.losses import CombinedSegmentationLoss
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.seed import set_seed
    from utils.logger import setup_logger
    from utils.checkpoint import save_checkpoint, load_checkpoint
    from models.swin import SwinSegmentation, replace_swin_ffn_with_moe
    from evaluation.metrics import compute_segmentation_metrics, SegmentationMetricAccumulator
    from evaluation.visualize_predictions import plot_segmentation_curves, plot_segmentation_predictions
    from evaluation.losses import CombinedSegmentationLoss

def train(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_cfg = config["model"]
    train_cfg = config["training"]
    moe_cfg = config["moe"]
    
    # Generate timestamp for unique output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("outputs", f"swin_moe_{timestamp}")
    log_dir = os.path.join(experiment_dir, "logs")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    plot_dir = os.path.join(experiment_dir, "plots")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set seed
    set_seed(train_cfg["seed"])
    
    # Setup logger
    logger = setup_logger("swin_moe", log_dir=log_dir)
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Experiment Directory: {experiment_dir}")
    
    epochs_to_run = train_cfg["num_epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Dataloaders
    logger.info("Initializing datasets and dataloaders...")
    try:
        from datasets.medical_dataset import get_medical_dataloaders
    except ImportError:
        from vision_transformer_research.datasets.medical_dataset import get_medical_dataloaders
        
    train_loader, val_loader, test_loader = get_medical_dataloaders(
        data_dir="data/medical_dataset",
        batch_size=train_cfg["batch_size"],
        img_size=train_cfg["img_size"],
        num_workers=2,
        seed=train_cfg["seed"]
    )
    
    # Model
    logger.info(f"Loading Swin Vanilla model: {model_cfg['backbone']}...")
    model = SwinSegmentation(
        model_name=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=True
    )
    
    # Inject MoE
    logger.info("Injecting Spatially-Aware MoE into Swin Transformer...")
    replaced_count = replace_swin_ffn_with_moe(model, moe_cfg, logger)
    logger.info(f"Replaced {replaced_count} FFN blocks with MoE layers.")
    
    model.to(device)
    
    # Print model params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
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
    
    scaler = torch.amp.GradScaler('cuda')
    
    tb_writer = SummaryWriter(log_dir=log_dir) if SummaryWriter is not None else None
        
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
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(images) 
                loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            total_pixels += images.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch}/{epochs_to_run}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / total_pixels
        train_losses.append(epoch_train_loss)
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        # Validation epoch
        model.eval()
        running_val_loss = 0.0
        val_pixels = 0
        
        val_accumulator = SegmentationMetricAccumulator(num_classes=model_cfg["num_classes"])
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images, targets = images.to(device), targets.to(device)
                
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, targets)
                
                running_val_loss += loss.item() * images.size(0)
                val_pixels += images.size(0)
                
                preds = logits.argmax(dim=1)
                val_accumulator.update(preds, targets)
                    
        epoch_val_loss = running_val_loss / val_pixels
        val_losses.append(epoch_val_loss)
        
        metrics = val_accumulator.compute()
        val_ious.append(metrics["mean_iou"])
        val_dices.append(metrics["mean_dice"])
        
        logger.info(
            f"Epoch [{epoch}/{epochs_to_run}] (LR: {current_lr:.2e}) - "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val mIoU: {metrics['mean_iou']:.4f} | "
            f"Val mDice: {metrics['mean_dice']:.4f}"
        )
        
        if tb_writer:
            tb_writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
            tb_writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
            tb_writer.add_scalar("Metrics/mIoU", metrics["mean_iou"], epoch)
            tb_writer.add_scalar("Metrics/mDice", metrics["mean_dice"], epoch)
            
        # Checkpoint save
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou": best_val_iou,
        }
        save_checkpoint(state, checkpoint_dir, filename="last_checkpoint.pth")
        
        if metrics["mean_iou"] > best_val_iou:
            best_val_iou = metrics["mean_iou"]
            patience_counter = 0
            save_checkpoint(state, checkpoint_dir, filename="best_model.pth")
            logger.info(f"New best validation mIoU: {best_val_iou:.4f}. Saved best model checkpoint.")
        else:
            patience_counter += 1
            logger.info(f"Validation mIoU did not improve. Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered. Training stopped.")
                break
                
    # Testing
    logger.info("Loading best model for testing...")
    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, device=device)
        
    model.eval()
    test_accumulator = SegmentationMetricAccumulator(num_classes=model_cfg["num_classes"])
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(images)
            preds = logits.argmax(dim=1)
            test_accumulator.update(preds, targets)
            
    test_metrics = test_accumulator.compute()
    logger.info(
        f"Test Set Metrics - "
        f"mIoU: {test_metrics['mean_iou']:.4f} | "
        f"mDice: {test_metrics['mean_dice']:.4f} | "
        f"Pixel Acc: {test_metrics['pixel_accuracy']:.4f} | "
        f"Precision: {test_metrics['mean_precision']:.4f} | "
        f"Recall: {test_metrics['mean_recall']:.4f}"
    )
    
    import json
    with open(os.path.join(log_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    
    if tb_writer:
        tb_writer.close()
    logger.info("Training process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin-MoE Semantic Segmentation")
    parser.add_argument("--config", type=str, default="configs/swin_segmentation.yaml", help="Path to config file")
    parser.add_argument("--warm-init", action="store_true", help="Enable warm initialization of MoE experts from pretrained FFN")
    args = parser.parse_args()
    
    # We can override the config dynamically
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.warm_init:
        config["moe"]["warm_init"] = True
        
    # Write temp config
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config = f.name
        
    train(temp_config)
    os.remove(temp_config)
