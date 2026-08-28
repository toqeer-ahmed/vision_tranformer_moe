import os
import sys
import yaml
import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.segmote import SegMoTE
from evaluation.losses import CombinedSegmentationLoss
from evaluation.metrics import compute_segmentation_metrics
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint

def train(config_path, fast_dev_run=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    train_cfg = config["training"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    log_cfg = config["logging"]
    
    set_seed(train_cfg["seed"])
    logger = setup_logger("segmote", log_dir=log_cfg["log_dir"])
    logger.info(f"Loaded config from {config_path}")
    
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Dataloaders
    logger.info("Initializing datasets and dataloaders...")
    if fast_dev_run:
        logger.warning("Using random mock data for fast development run.")
        mock_images = torch.randn(2, 3, 512, 512) # SAM can process 512x512
        mock_masks = torch.randint(0, model_cfg["num_classes"], (2, 512, 512))
        mock_dataset = torch.utils.data.TensorDataset(mock_images, mock_masks)
        train_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=2, shuffle=True)
        val_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=2, shuffle=False)
        test_loader = val_loader
    else:
        # Load dataset
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
        
    # Model
    logger.info("Loading SegMoTE (SAM + MoTE + PPT)...")
    # By default, use facebook/sam-vit-base
    model = SegMoTE(sam_model_name="facebook/sam-vit-base", num_experts=4, top_k=2)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Loss Setup (Includes LoadBalancingLoss logic inside CombinedSegmentationLoss)
    loss_type = train_cfg.get("loss_type", "focal_tversky")
    dice_weight = float(train_cfg.get("dice_weight", 1.0))
    criterion = CombinedSegmentationLoss(loss_type=loss_type, dice_weight=dice_weight).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"])
    )
    
    epochs = train_cfg["epochs"] if not fast_dev_run else 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    tb_writer = SummaryWriter(log_dir=log_cfg["log_dir"]) if log_cfg["use_tensorboard"] else None
    
    best_val_iou = 0.0
    
    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = dataset_cfg.get("accumulation_steps", 4)
    
    logger.info("Starting SegMoTE training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Forward with AMP
            with torch.cuda.amp.autocast():
                logits, gating_probs = model(images)
                loss = criterion(logits, targets, gating_probs)
                loss = loss / accumulation_steps
            
            # Backward
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += (loss.item() * accumulation_steps)
            
            if batch_idx % log_cfg["log_interval"] == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item() * accumulation_steps:.4f}")
                
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss, val_iou, val_dice = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    logits, gating_probs = model(images)
                    loss = criterion(logits, targets, gating_probs)
                
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).long()
                metrics = compute_segmentation_metrics(preds, targets)
                val_iou += metrics["mean_iou"]
                val_dice += metrics["mean_dice"]
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        logger.info(f"Epoch [{epoch}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_iou:.4f} | Val mDice: {avg_val_dice:.4f}")
        
        if tb_writer:
            tb_writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            tb_writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            tb_writer.add_scalar("Metrics/mIoU", avg_val_iou, epoch)
            tb_writer.add_scalar("Metrics/mDice", avg_val_dice, epoch)
            
        scheduler.step()
        
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            logger.info(f"New best validation mIoU: {best_val_iou:.4f}. Saving checkpoint...")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"mIoU": best_val_iou},
                checkpoint_dir=log_cfg["checkpoint_dir"],
                filename="segmote_best.pth"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/medical_segmentation.yaml")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run 2 epochs on mock data")
    args = parser.parse_args()
    
    train(args.config, fast_dev_run=args.fast_dev_run)
