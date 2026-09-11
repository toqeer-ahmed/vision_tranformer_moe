import os
import sys

# Ensure the root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_exp_b import train as train_exp_b
from training.train_exp_d import train as train_exp_d
from utils.logger import setup_logger

def run_unified_experiment():
    # Set up a master logger
    logger = setup_logger("ablation_phase2", log_dir="outputs/ablation_phase2")
    logger.info("=========================================================")
    logger.info("Starting MoE Ablation Phase 2 (BUSI Dataset)")
    logger.info("Comparing: Exp B (MLP No-Conv) vs. Exp D (Spatial MoE)")
    logger.info("Data Split: Strict 80% Train, 10% Validation, 10% Test")
    logger.info("=========================================================")
    
    # 1. Train Vanilla SegFormer
    logger.info("\n")
    logger.info("---------------------------------------------------------")
    logger.info("PHASE 1: Training Experiment B (MLP Baseline, No Conv, No MoE)")
    logger.info("---------------------------------------------------------")
    try:
        # Reusing the vanilla config structure, but train_exp_b will dynamically replace the MixFFN with SimpleMLPBlock
        train_exp_b("configs/vanilla_segmentation.yaml")
        logger.info("Experiment B training and testing completed successfully.")
    except Exception as e:
        logger.error(f"Experiment B training failed: {e}")
        return

    # 2. Train MoE-SegFormer
    logger.info("\n")
    logger.info("---------------------------------------------------------")
    logger.info("PHASE 2: Training Experiment D (Spatially-Aware MoE)")
    logger.info("---------------------------------------------------------")
    try:
        # Reusing moe config structure, but train_exp_d will use DenseMoELayer with SpatiallyAwareExpert
        train_exp_d("configs/moe_segmentation.yaml")
        logger.info("Experiment D training and testing completed successfully.")
    except Exception as e:
        logger.error(f"Experiment D training failed: {e}")
        return

    logger.info("\n")
    logger.info("=========================================================")
    logger.info("MoE Ablation Phase 2 Completed Successfully!")
    logger.info("Please zip and download the 'outputs' folder for analysis.")
    logger.info("=========================================================")

if __name__ == "__main__":
    run_unified_experiment()
