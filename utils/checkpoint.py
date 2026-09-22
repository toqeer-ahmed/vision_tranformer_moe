import os
import torch
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth"
):
    """
    Saves the training state (model, optimizer, scheduler, epoch, etc.) to a file.
    
    Args:
        state (dict): State dictionary containing model state_dict, optimizer state_dict, etc.
        checkpoint_dir (str): Directory where the checkpoint should be saved.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
    device: str = "cpu"
) -> dict:
    """
    Loads training state from checkpoint path.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The learning rate scheduler to load state into.
        device (str): Device to map checkpoint to ('cpu', 'cuda', etc.).
        
    Returns:
        dict: The raw checkpoint dictionary.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    # Load optimizer state dict
    if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        
    # Load scheduler state dict
    if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
        
    logger.info("Checkpoint loaded successfully.")
    return checkpoint
