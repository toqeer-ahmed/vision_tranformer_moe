import os
import sys
import yaml
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.medical_dataset import MedicalImageMaskDataset, get_medical_transforms
from evaluation.metrics import compute_segmentation_metrics, SegmentationMetricAccumulator

from models.segformer import SegFormerSegmentation
from models.segmote import SegMoTE
from training.train_exp_f import replace_segformer_ffn_with_wide_ffn
from training.train_exp_e import replace_segformer_ffn_with_moe
from models.swin import SwinSegmentation, replace_swin_ffn_with_moe
from utils.logger import setup_logger

def load_model(config_path, checkpoint_path, device, model_name_key="vanilla"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_cfg = config['model']
    # Infer architecture type
    arch_type = model_cfg.get('type', None)
    if arch_type is None:
        if 'backbone' in model_cfg or 'swin' in model_name_key.lower():
            arch_type = 'swin'
        else:
            arch_type = 'segformer'
            
    if arch_type == 'swin':
        model = SwinSegmentation(
            model_name=model_cfg['backbone'],
            num_classes=model_cfg['num_classes'],
            pretrained=False
        )
        if "moe" in model_name_key.lower():
            dummy_logger = setup_logger("dummy", log_dir="/tmp")
            replace_swin_ffn_with_moe(model, config.get("moe", {}), dummy_logger)
            
    elif arch_type == 'segformer':
        model = SegFormerSegmentation(
            model_name=model_cfg['name'],
            num_classes=model_cfg['num_classes'],
            pretrained=False
        )
        if "moe" in model_name_key.lower():
            dummy_logger = setup_logger("dummy", log_dir="/tmp")
            replace_segformer_ffn_with_moe(model, config.get("moe", {}), dummy_logger)
        elif "wide" in model_name_key.lower():
            dummy_logger = setup_logger("dummy", log_dir="/tmp")
            replace_segformer_ffn_with_wide_ffn(model, config.get("moe", {}), dummy_logger)
            
    elif arch_type == 'segmote':
        model = SegMoTE(
            model_name=model_cfg['name'],
            num_classes=model_cfg['num_classes'],
            num_experts=model_cfg.get('num_experts', 4),
            top_k=model_cfg.get('top_k', 2),
            expert_type=model_cfg.get('expert_type', 'spatial')
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model.to(device)
    model.eval()
    return model

def reset_moe_stats(model):
    for name, module in model.named_modules():
        if hasattr(module, 'expert_token_counts'):
            module.expert_token_counts.zero_()
        if hasattr(module, 'eval_routing_entropy'):
            module.eval_routing_entropy.zero_()
        if hasattr(module, 'eval_total_tokens'):
            module.eval_total_tokens.zero_()

def extract_moe_stats(model):
    total_entropy = 0.0
    total_tokens = 0
    expert_counts = None
    
    for name, module in model.named_modules():
        if hasattr(module, 'expert_token_counts'):
            if expert_counts is None:
                expert_counts = module.expert_token_counts.clone()
            else:
                expert_counts += module.expert_token_counts
                
        if hasattr(module, 'eval_routing_entropy'):
            total_entropy += module.eval_routing_entropy.item()
            total_tokens += module.eval_total_tokens.item()
            
    if total_tokens > 0:
        mean_entropy = total_entropy / total_tokens
    else:
        mean_entropy = 0.0
        
    if expert_counts is not None:
        total_assignments = expert_counts.sum().item()
        if total_assignments > 0:
            utilization = (expert_counts / total_assignments).cpu().tolist()
        else:
            utilization = []
    else:
        utilization = []
        
    return {
        "mean_routing_entropy": mean_entropy,
        "expert_utilization": utilization
    }

def evaluate_zero_shot(dataset_dir, checkpoints, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger("zero_shot_eval", log_dir=output_dir)
    
    logger.info("Initializing zero-shot cross-dataset evaluation on BUS-BRA.")
    
    # 1. Prepare Dataset (All 1875 images, no splitting)
    _, val_transform = get_medical_transforms(img_size=352)
    dataset = MedicalImageMaskDataset(
        data_dir=dataset_dir,
        transform=val_transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    logger.info(f"Loaded full dataset: {len(dataset)} images.")
    
    # Generate random indices for visualization
    np.random.seed(42)
    vis_indices = np.random.choice(len(dataset), size=5, replace=False).tolist()
    
    results = {}
    
    for model_name, paths in checkpoints.items():
        logger.info(f"Evaluating {model_name}...")
        model = load_model(paths['config'], paths['checkpoint'], device, model_name_key=model_name)
        reset_moe_stats(model)
        
        eval_accumulator = SegmentationMetricAccumulator(num_classes=2)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc=model_name)):
                images = images.to(device)
                masks = masks.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                # Ensure outputs are upsampled to mask size if necessary
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    
                preds = torch.argmax(outputs, dim=1)
                eval_accumulator.update(preds, masks)
                
        metrics = eval_accumulator.compute()
        
        bg_iou = metrics["class_iou"][0]
        fg_iou = metrics["class_iou"][1]
        bg_dice = metrics["class_dice"][0]
        fg_dice = metrics["class_dice"][1]
        
        # Calculate derived metrics from confusion matrix components
        tp = metrics["class_tp"][1]
        fp = metrics["class_fp"][1]
        fn = metrics["class_fn"][1]
        tn = metrics["class_tp"][0]  # True negatives for foreground is True positives for background
        
        fg_precision = tp / (tp + fp + 1e-8)
        fg_recall = tp / (tp + fn + 1e-8)
        
        logger.info(f"[{model_name}] Foreground (Lesion) IoU: {fg_iou*100:.2f}% | Dice: {fg_dice*100:.2f}% | Precision: {fg_precision*100:.2f}% | Recall: {fg_recall*100:.2f}%")
        logger.info(f"[{model_name}] Background IoU: {bg_iou*100:.2f}% | Dice: {bg_dice*100:.2f}%")
        logger.info(f"[{model_name}] Overall mIoU: {metrics['mean_iou']*100:.2f}% | mDice: {metrics['mean_dice']*100:.2f}% | Pixel Acc: {metrics['pixel_accuracy']*100:.2f}%")
        logger.info(f"[{model_name}] Confusion Matrix -> TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
        
        # Add MoE stats if applicable
        if "moe" in model_name.lower():
            moe_stats = extract_moe_stats(model)
            metrics.update(moe_stats)
            logger.info(f"[{model_name}] Routing Entropy: {moe_stats['mean_routing_entropy']:.4f}")
            logger.info(f"[{model_name}] Expert Utilization: {moe_stats['expert_utilization']}")
            
        results[model_name] = metrics
        
        # Save visualization samples
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(vis_indices), 3, figsize=(12, 4 * len(vis_indices)))
        if len(vis_indices) == 1: axes = [axes]
        
        for row_idx, idx in enumerate(vis_indices):
            img_tensor, mask_tensor = dataset[idx]
            # De-normalize image for visualization
            img_np = img_tensor.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            with torch.no_grad():
                out = model(img_tensor.unsqueeze(0).to(device))
                if out.shape[-2:] != mask_tensor.shape:
                    out = torch.nn.functional.interpolate(out, size=mask_tensor.shape, mode='bilinear', align_corners=False)
                pred_np = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
                
            axes[row_idx][0].imshow(img_np)
            axes[row_idx][0].set_title(f"Image {idx}")
            axes[row_idx][0].axis('off')
            
            axes[row_idx][1].imshow(mask_tensor.numpy(), cmap='gray')
            axes[row_idx][1].set_title("Ground Truth")
            axes[row_idx][1].axis('off')
            
            axes[row_idx][2].imshow(pred_np, cmap='gray')
            axes[row_idx][2].set_title("Prediction")
            axes[row_idx][2].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_vis.png"))
        plt.close()

    # Save all results
    with open(os.path.join(output_dir, "zero_shot_metrics.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info("Evaluation complete. Results saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/busbra_processed")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/zero_shot_results")
    parser.add_argument("--vanilla_ckpt", type=str, required=True)
    parser.add_argument("--moe_ckpt", type=str, required=True)
    parser.add_argument("--wide_ckpt", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoints = {
        "vanilla": {
            "config": "configs/vanilla_segmentation.yaml",
            "checkpoint": args.vanilla_ckpt
        },
        "moe_warm_init": {
            "config": "configs/moe_segmentation.yaml",
            "checkpoint": args.moe_ckpt
        },
        "wide_control": {
            "config": "configs/moe_segmentation.yaml", # Wide control uses moe config but with expansion 16 usually? Wait, the config needs to specify wide!
            "checkpoint": args.wide_ckpt
        }
    }
    
    evaluate_zero_shot(args.data_dir, checkpoints, args.output_dir)
