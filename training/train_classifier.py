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
    from vision_transformer_research.datasets.classification_dataset import get_classification_dataloaders
    from vision_transformer_research.models.vit_classifier import ViTClassifier
    from vision_transformer_research.evaluation.metrics import compute_classification_metrics
    from vision_transformer_research.evaluation.visualize_predictions import plot_classification_curves, plot_confusion_matrix
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.seed import set_seed
    from utils.logger import setup_logger
    from utils.checkpoint import save_checkpoint, load_checkpoint
    from datasets.classification_dataset import get_classification_dataloaders
    from models.vit_classifier import ViTClassifier
    from evaluation.metrics import compute_classification_metrics
    from evaluation.visualize_predictions import plot_classification_curves, plot_confusion_matrix

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
    logger = setup_logger("classification", log_dir=log_cfg["log_dir"])
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
    
    # Modify classes count if CIFAR-100
    num_classes = 100 if dataset_cfg["name"].lower() == "cifar100" else 10

    # Dataloaders
    logger.info("Initializing datasets and dataloaders...")
    if fast_dev_run:
        logger.warning("Using random mock data for fast development run to bypass slow dataset downloads.")
        mock_images = torch.randn(10, 3, model_cfg["img_size"], model_cfg["img_size"])
        mock_targets = torch.randint(0, num_classes, (10,))
        mock_dataset = torch.utils.data.TensorDataset(mock_images, mock_targets)
        train_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=dataset_cfg["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=dataset_cfg["batch_size"], shuffle=False)
        test_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=dataset_cfg["batch_size"], shuffle=False)
    else:
        train_loader, val_loader, test_loader = get_classification_dataloaders(
            dataset_name=dataset_cfg["name"],
            data_dir=dataset_cfg["data_dir"],
            batch_size=dataset_cfg["batch_size"],
            img_size=model_cfg["img_size"],
            num_workers=dataset_cfg["num_workers"],
            seed=train_cfg["seed"]
        )
    
    # Model
    logger.info(f"Loading ViT model: {model_cfg['name']}...")
    model = ViTClassifier(
        model_name=model_cfg["name"],
        num_classes=num_classes,
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
    train_accs, val_accs = [], []
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping_patience"]
    
    logger.info("Starting training loop...")
    for epoch in range(1, epochs_to_run + 1):
        # Training epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Fast development run limits batches
            if fast_dev_run and batch_idx >= 2:
                break
                
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            if (batch_idx + 1) % 50 == 0 or fast_dev_run:
                logger.info(f"Epoch [{epoch}/{epochs_to_run}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation epoch
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                if fast_dev_run and batch_idx >= 2:
                    break
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                
                running_val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                
        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        logger.info(f"Epoch [{epoch}/{epochs_to_run}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Tensorboard log
        if tb_writer:
            tb_writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
            tb_writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
            tb_writer.add_scalar("Accuracy/Train", epoch_train_acc, epoch)
            tb_writer.add_scalar("Accuracy/Val", epoch_val_acc, epoch)
            
        # Checkpoint save
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_val_acc,
        }
        save_checkpoint(state, log_cfg["checkpoint_dir"], filename="last_checkpoint.pth")
        
        if epoch_val_acc > best_val_acc or fast_dev_run:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            save_checkpoint(state, log_cfg["checkpoint_dir"], filename="best_model.pth")
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}. Saved best model checkpoint.")
        else:
            patience_counter += 1
            logger.info(f"Validation accuracy did not improve. Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered. Training stopped.")
                break
                
    # Plot curves
    plot_classification_curves(train_losses, val_losses, train_accs, val_accs, log_cfg["plot_dir"])
    
    # Testing
    logger.info("Loading best model for testing...")
    best_path = os.path.join(log_cfg["checkpoint_dir"], "best_model.pth")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, device=device)
        
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if fast_dev_run and batch_idx >= 2:
                break
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = compute_classification_metrics(all_preds, all_targets)
    logger.info(f"Test Set Metrics - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1_score']:.4f}")
    
    # Class names
    if dataset_cfg["name"].lower() == "cifar10":
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        classes = [str(i) for i in range(100)]
        
    # Plot Confusion Matrix
    cm_path = os.path.join(log_cfg["plot_dir"], "confusion_matrix.png")
    # In fast_dev_run, confusion matrix might have missing predictions for classes, so adjust target/pred classes
    active_classes = [classes[i] for i in sorted(list(set(all_targets) | set(all_preds)))]
    # Standard compute confusion_matrix will return only shape for classes present if not specified, 
    # but compute_classification_metrics computes it on raw labels. Let's make sure class names length match cm size:
    cm_to_plot = metrics["confusion_matrix"]
    plot_confusion_matrix(cm_to_plot, active_classes, cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    if tb_writer:
        tb_writer.close()
        
    logger.info("Training process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer Classifier")
    parser.add_argument("--config", type=str, default="configs/classification.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)
