import os
import sys

# Ensure the root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_segmentation import train as train_vanilla
from training.train_moe import train as train_moe
from utils.logger import setup_logger

def run_unified_experiment():
    # Set up a master logger
    logger = setup_logger("unified_experiment", log_dir="outputs/unified_comparison")
    logger.info("=========================================================")
    logger.info("Starting Unified Controlled Experiment (BUSI Dataset)")
    logger.info("Comparing: Vanilla SegFormer-B0 vs. MoE-SegFormer")
    logger.info("Data Split: Strict 80% Train, 10% Validation, 10% Test")
    logger.info("=========================================================")
    
    # 1. Train Vanilla SegFormer
    logger.info("\n")
    logger.info("---------------------------------------------------------")
    logger.info("PHASE 1: Training Vanilla SegFormer-B0 Baseline")
    logger.info("---------------------------------------------------------")
    try:
        train_vanilla("configs/vanilla_segmentation.yaml")
        logger.info("Vanilla SegFormer training and testing completed successfully.")
    except Exception as e:
        logger.error(f"Vanilla SegFormer training failed: {e}")
        return

    # 2. Train MoE-SegFormer
    logger.info("\n")
    logger.info("---------------------------------------------------------")
    logger.info("PHASE 2: Training MoE-SegFormer (Proposed Model)")
    logger.info("---------------------------------------------------------")
    try:
        train_moe("configs/moe_segmentation.yaml")
        logger.info("MoE-SegFormer training and testing completed successfully.")
    except Exception as e:
        logger.error(f"MoE-SegFormer training failed: {e}")
        return

    logger.info("\n")
    logger.info("=========================================================")
    logger.info("Unified Controlled Experiment Completed Successfully!")
    logger.info("Please zip and download the 'outputs' folder for analysis.")
    logger.info("=========================================================")

if __name__ == "__main__":
    run_unified_experiment()
