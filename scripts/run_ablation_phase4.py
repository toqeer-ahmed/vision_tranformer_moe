import os
import sys

# Ensure the root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_exp_f import train as train_exp_f
from utils.logger import setup_logger

def run_unified_experiment():
    # Set up a master logger
    logger = setup_logger("ablation_phase4", log_dir="outputs/ablation_phase4")
    logger.info("=========================================================")
    logger.info("Starting MoE Ablation Phase 4 (Fair Non-MoE Control)")
    logger.info("Comparing: Exp E (Warm Init Spatial MoE) vs. Exp F (Wide SegFormer Control)")
    logger.info("Data Split: Strict 80% Train, 10% Validation, 10% Test")
    logger.info("=========================================================")
    
    # 1. Train Wide SegFormer Control
    logger.info("\n")
    logger.info("---------------------------------------------------------")
    logger.info("PHASE 4: Training Experiment F (Wide SegFormer Control with Warm Tiling)")
    logger.info("---------------------------------------------------------")
    try:
        # We reuse the moe config, but train_exp_f intercepts and injects the wide tiled blocks
        train_exp_f("configs/moe_segmentation.yaml")
        logger.info("Experiment F training and testing completed successfully.")
    except Exception as e:
        logger.error(f"Experiment F training failed: {e}")
        return

    logger.info("\n")
    logger.info("=========================================================")
    logger.info("MoE Ablation Phase 4 Completed Successfully!")
    logger.info("Please zip and download the 'outputs' folder for analysis.")
    logger.info("=========================================================")

if __name__ == "__main__":
    run_unified_experiment()
