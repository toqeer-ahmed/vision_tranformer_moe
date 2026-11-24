import unittest
import torch
import torch.nn as nn
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.unet import UNet
from evaluation.losses import CombinedSegmentationLoss
from evaluation.metrics import SegmentationMetricAccumulator

class TestStandardUNet(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=3, num_classes=3, base_channels=64, bilinear=False).to(self.device)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        self.target_tensor = torch.randint(0, 3, (self.batch_size, 224, 224), device=self.device).long()

    def test_01_parameter_count(self):
        """Verify parameter counts programmatically across components."""
        counts = self.model.count_parameters()
        total = counts["total_parameters"]
        trainable = counts["trainable_parameters"]
        encoder = counts["encoder_parameters"]
        bottleneck = counts["bottleneck_parameters"]
        decoder = counts["decoder_parameters"]
        head = counts["head_parameters"]
        
        print("\n=== U-NET PARAMETER COUNT AUDIT ===")
        print(f"Encoder Parameters:    {encoder:,}")
        print(f"Bottleneck Parameters: {bottleneck:,}")
        print(f"Decoder Parameters:    {decoder:,}")
        print(f"Head Parameters:       {head:,}")
        print(f"Total Parameters:      {total:,}")
        print(f"Trainable Parameters:  {trainable:,}")
        
        self.assertEqual(total, trainable, "All parameters in standard U-Net must be trainable (no frozen layers)")
        self.assertEqual(total, encoder + bottleneck + decoder + head, "Component sums must equal total")
        self.assertGreater(total, 30_000_000, "Canonical 64-base U-Net should have ~31.04M parameters")
        self.assertLess(total, 32_000_000, "Canonical 64-base U-Net should have ~31.04M parameters")

    def test_02_forward_pass_dimensions(self):
        """Verify that output dimensions strictly match (B, num_classes, H, W)."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input_tensor)
        expected_shape = (self.batch_size, 3, 224, 224)
        self.assertEqual(output.shape, expected_shape, f"Expected {expected_shape}, got {output.shape}")
        self.assertFalse(torch.isnan(output).any(), "Output tensor contains NaN values")

    def test_03_loss_and_backward_pass(self):
        """Verify Focal Tversky Loss calculation and full backpropagation gradient flow."""
        self.model.train()
        criterion = CombinedSegmentationLoss(loss_type="focal_tversky", dice_weight=1.0).to(self.device)
        
        logits = self.model(self.input_tensor)
        loss = criterion(logits, self.target_tensor)
        
        self.assertFalse(torch.isnan(loss), "Loss is NaN")
        self.assertGreater(loss.item(), 0.0, "Loss must be positive")
        
        loss.backward()
        
        # Verify gradients exist and are non-zero across encoder, bottleneck, and decoder
        has_grad_count = 0
        total_param_count = 0
        for name, param in self.model.named_parameters():
            total_param_count += 1
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient after backward")
                grad_norm = param.grad.data.norm().item()
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN")
                if grad_norm > 0:
                    has_grad_count += 1
                    
        self.assertEqual(has_grad_count, total_param_count, "All trainable parameters must receive non-zero gradients")
        print(f"Verified active gradients for all {total_param_count} parameter tensors.")

    def test_04_checkpoint_save_and_load(self):
        """Verify checkpoint state_dict serialization and load parity."""
        os.makedirs("scratch", exist_ok=True)
        ckpt_path = "scratch/test_unet_ckpt.pth"
        
        # Save
        torch.save(self.model.state_dict(), ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path), "Checkpoint file was not created")
        
        # Load into new model
        new_model = UNet(in_channels=3, num_classes=3, base_channels=64, bilinear=False).to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        new_model.load_state_dict(state_dict)
        
        # Verify predictions match identically
        self.model.eval()
        new_model.eval()
        with torch.no_grad():
            out1 = self.model(self.input_tensor)
            out2 = new_model(self.input_tensor)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6), "Loaded model outputs do not match original")
        
        # Cleanup
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    def test_05_metric_accumulator(self):
        """Verify that SegmentationMetricAccumulator processes U-Net outputs accurately."""
        acc = SegmentationMetricAccumulator(num_classes=3)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.input_tensor)
            preds = logits.argmax(dim=1)
        acc.update(preds, self.target_tensor)
        metrics = acc.compute()
        
        for k in ["pixel_accuracy", "mean_iou", "mean_dice", "mean_precision", "mean_recall"]:
            self.assertIn(k, metrics, f"Missing metric {k}")
            self.assertTrue(0.0 <= metrics[k] <= 1.0, f"Metric {k} out of range: {metrics[k]}")
        print("Metric accumulator verification successful.")

if __name__ == "__main__":
    unittest.main()
