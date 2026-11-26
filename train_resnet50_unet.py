import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from training.train_resnet50_unet import train
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-50 U-Net Baseline on BUSI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    train(args.config)
