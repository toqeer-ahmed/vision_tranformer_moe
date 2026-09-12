import os
import sys

# Ensure the root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_exp_e import train as train_exp_e
from utils.logger import setup_logger

def run_unified_experiment():
    # Set up a master logger
    logger = setup_logger("ablation_phase3", log_dir="outputs/ablation_phase3")
    logger.info("=========================================================")
    logger.info("Starting MoE Ablation Phase 3 (Warm Initialization)")
    logger.info("Comparing: Exp D (Random Init Spatial MoE) vs. Exp E (Warm Init Spatial MoE)")
    logger.info("Data Split: Strict 80% Train, 10% Validation, 10% Test")
    logger.info("=========================================================")
    
    # 1. Train Warm Initialized MoE
    logger.info("\n")
    logger.info("---------------------------------------------------------")
    logger.info("PHASE 3: Training Experiment E (Warm Initialized Spatially-Aware MoE)")
    logger.info("---------------------------------------------------------")
    try:
        # We reuse the moe config, but train_exp_e intercepts and injects the pre-trained weights
        train_exp_e("configs/moe_segmentation.yaml")
        logger.info("Experiment E training and testing completed successfully.")
    except Exception as e:
        logger.error(f"Experiment E training failed: {e}")
        return

    logger.info("\n")
    logger.info("=========================================================")
    logger.info("MoE Ablation Phase 3 Completed Successfully!")
    logger.info("Please zip and download the 'outputs' folder for analysis.")
    logger.info("=========================================================")

if __name__ == "__main__":
    run_unified_experiment()
