import os
import argparse
import yaml
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from vision_transformer_research.models.vit_classifier import ViTClassifier
    from vision_transformer_research.models.segformer import SegFormerSegmentation
    from vision_transformer_research.models.moe.moe_layer import MoELayer
    from vision_transformer_research.utils.checkpoint import load_checkpoint
    from vision_transformer_research.training.train_moe import replace_segformer_ffn_with_moe
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.vit_classifier import ViTClassifier
    from models.segformer import SegFormerSegmentation
    from models.moe.moe_layer import MoELayer
    from utils.checkpoint import load_checkpoint
    from training.train_moe import replace_segformer_ffn_with_moe

def predict_classification(model, image_path, img_size, device, class_names):
    """
    Inference for classification. Prints and returns predicted label and probability.
    """
    image = Image.open(image_path).convert("RGB")
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ])
    image_np = np.array(image)
    transformed = transform(image=image_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        prob = probs[0, pred_idx].item()
        
    class_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
    print(f"\n--- Inference Result ---")
    print(f"Prediction: {class_label}")
    print(f"Confidence score: {prob:.4f}")
    return class_label, prob

def predict_segmentation(model, image_path, img_size, device, save_path):
    """
    Inference for semantic segmentation. Saves visual comparison plot of input vs mask.
    """
    image = Image.open(image_path).convert("RGB")
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image_np = np.array(image)
    transformed = transform(image=image_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
    # Prepare comparison plot
    img_disp = np.clip(np.array(image.resize((img_size, img_size))) / 255.0, 0, 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_disp)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(preds, cmap="gray")
    axes[1].set_title("Predicted Segmentation Mask")
    axes[1].axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"\n--- Inference Result ---")
    print(f"Segmentation mask visual comparison saved to: {save_path}")
    return preds

def main():
    parser = argparse.ArgumentParser(description="Vision Transformer Research Framework Inference Script")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "segmentation", "moe_segmentation"], help="Task mode")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--save_path", type=str, default="outputs/predictions/prediction_result.png", help="Path to save result plot")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.task == "classification":
        num_classes = 100 if dataset_cfg["name"].lower() == "cifar100" else 10
        model = ViTClassifier(
            model_name=model_cfg["name"],
            num_classes=num_classes,
            pretrained=False
        )
        load_checkpoint(args.checkpoint, model, device=device)
        model.to(device)
        
        # CIFAR-10 classes
        if dataset_cfg["name"].lower() == "cifar10":
            class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        else:
            class_names = [str(i) for i in range(100)]
            
        predict_classification(model, args.image, model_cfg["img_size"], device, class_names)
        
    elif args.task in ["segmentation", "moe_segmentation"]:
        model = SegFormerSegmentation(
            model_name=model_cfg["name"],
            num_classes=model_cfg["num_classes"],
            pretrained=False
        )
        
        if args.task == "moe_segmentation":
            moe_cfg = model_cfg["moe"]
            import logging
            # Set up logger for MoE block replacement logging
            logger = logging.getLogger("moe_segmentation")
            logging.basicConfig(level=logging.INFO)
            replace_segformer_ffn_with_moe(model, moe_cfg, logger)
            
        load_checkpoint(args.checkpoint, model, device=device)
        model.to(device)
        
        predict_segmentation(model, args.image, model_cfg["img_size"], device, args.save_path)

if __name__ == "__main__":
    main()
