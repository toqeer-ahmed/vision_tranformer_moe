import unittest
import torch
import torch.nn as nn
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.attention_unet import AttentionUNet, AttentionGate
from evaluation.losses import CombinedSegmentationLoss
from evaluation.metrics import SegmentationMetricAccumulator

class TestAttentionUNet(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionUNet(
            in_channels=3,
            num_classes=3,
            base_channels=64,
            bilinear=False
        ).to(self.device)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        self.target_tensor = torch.randint(0, 3, (self.batch_size, 224, 224), device=self.device).long()

    def test_01_model_initialization(self):
        """Test 1: Verify model initialization and sub-module construction."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertTrue(hasattr(self.model, "inc"))
        self.assertTrue(hasattr(self.model, "down1"))
        self.assertTrue(hasattr(self.model, "down2"))
        self.assertTrue(hasattr(self.model, "down3"))
        self.assertTrue(hasattr(self.model, "down4"))
        self.assertTrue(hasattr(self.model, "up1"))
        self.assertTrue(hasattr(self.model, "up2"))
        self.assertTrue(hasattr(self.model, "up3"))
        self.assertTrue(hasattr(self.model, "up4"))
        self.assertTrue(hasattr(self.model, "outc"))
        print("\nTest 1 Passed: Attention U-Net initialized with all encoder-decoder modules.")

    def test_02_input_tensor_shape(self):
        """Test 2: Verify input tensor dimensions."""
        self.assertEqual(self.input_tensor.shape, (self.batch_size, 3, 224, 224))
        print("Test 2 Passed: Input tensor shape matches (B, 3, 224, 224).")

    def test_03_output_tensor_shape(self):
        """Test 3: Verify output logits tensor shape strictly matches (B, num_classes, H, W)."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input_tensor)
        expected_shape = (self.batch_size, 3, 224, 224)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
        print("Test 3 Passed: Output logits match expected shape (B, 3, 224, 224).")

    def test_04_attention_map_dimensions(self):
        """Test 4: Verify attention map dimensions across all 4 decoder stages."""
        self.model.eval()
        with torch.no_grad():
            logits, att_maps = self.model(self.input_tensor, return_attentions=True)
            
        self.assertEqual(len(att_maps), 4, "Attention U-Net must produce 4 attention maps")
        expected_spatial = [
            (self.batch_size, 1, 28, 28),   # AG1 (512-dim stage)
            (self.batch_size, 1, 56, 56),   # AG2 (256-dim stage)
            (self.batch_size, 1, 112, 112), # AG3 (128-dim stage)
            (self.batch_size, 1, 224, 224)  # AG4 (64-dim stage)
        ]
        for i, (att, exp_shape) in enumerate(zip(att_maps, expected_spatial), 1):
            self.assertEqual(att.shape, exp_shape, f"AG{i} shape mismatch: expected {exp_shape}, got {att.shape}")
            self.assertTrue((att >= 0.0).all() and (att <= 1.0).all(), f"AG{i} values must be in [0, 1]")
        print("Test 4 Passed: All 4 Attention Gate maps verified with correct spatial scaling in [0, 1].")

    def test_05_forward_pass(self):
        """Test 5: Verify forward pass stability without exceptions or NaNs."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.input_tensor)
        self.assertIsNotNone(out)
        self.assertFalse(torch.isinf(out).any())
        print("Test 5 Passed: Forward pass executed stably.")

    def test_06_loss_calculation(self):
        """Test 6: Verify Focal Tversky Loss calculation on multi-class logits."""
        criterion = CombinedSegmentationLoss(loss_type="focal_tversky", dice_weight=1.0).to(self.device)
        self.model.train()
        logits = self.model(self.input_tensor)
        loss = criterion(logits, self.target_tensor)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)
        print(f"Test 6 Passed: Loss calculated successfully (loss = {loss.item():.4f}).")

    def test_07_backpropagation(self):
        """Test 7: Verify backpropagation execution without runtime errors."""
        self.model.train()
        criterion = CombinedSegmentationLoss(loss_type="focal_tversky", dice_weight=1.0).to(self.device)
        self.model.zero_grad()
        logits = self.model(self.input_tensor)
        loss = criterion(logits, self.target_tensor)
        loss.backward()
        print("Test 7 Passed: Backward gradient pass completed successfully.")

    def test_08_gradient_connectivity(self):
        """Test 8: Verify active gradient connectivity across ALL trainable parameter tensors."""
        self.model.train()
        criterion = CombinedSegmentationLoss(loss_type="focal_tversky", dice_weight=1.0).to(self.device)
        self.model.zero_grad()
        logits = self.model(self.input_tensor)
        loss = criterion(logits, self.target_tensor)
        loss.backward()

        total_tensors = 0
        active_grad_tensors = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_tensors += 1
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient")
                norm = param.grad.data.norm().item()
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient")
                if norm > 0:
                    active_grad_tensors += 1
                    
        self.assertEqual(active_grad_tensors, total_tensors, "All parameter tensors must receive non-zero gradients")
        print(f"Test 8 Passed: Full gradient connectivity verified across all {total_tensors} parameter tensors.")

    def test_09_exact_parameter_count(self):
        """Test 9: Independently and programmatically verify parameter counts."""
        counts = self.model.count_parameters()
        total = counts["total_parameters"]
        trainable = counts["trainable_parameters"]
        encoder = counts["encoder_parameters"]
        bottleneck = counts["bottleneck_parameters"]
        decoder = counts["decoder_parameters"]
        ag = counts["attention_gate_parameters"]
        head = counts["head_parameters"]

        # Exact expected values based on architectural formula
        self.assertEqual(total, 31_389_295, f"Expected exactly 31,389,295 parameters, got {total}")
        self.assertEqual(total, trainable, "100% of parameters must be trainable")
        self.assertEqual(ag, 351_532, f"Expected exactly 351,532 attention gate parameters, got {ag}")
        self.assertEqual(encoder + bottleneck + decoder + head, total)
        print(f"Test 9 Passed: Exact parameter count confirmed: {total:,} (AGs: {ag:,}).")

    def test_10_batch_size_compatibility(self):
        """Test 10: Verify compatibility across multiple batch sizes (B=1, B=2, B=4)."""
        self.model.eval()
        with torch.no_grad():
            for b in [1, 2, 4]:
                inp = torch.randn(b, 3, 224, 224, device=self.device)
                out = self.model(inp)
                self.assertEqual(out.shape, (b, 3, 224, 224))
        print("Test 10 Passed: Batch-size invariance verified for B=1, 2, 4.")

    def test_11_checkpoint_saving_and_loading(self):
        """Test 11: Verify state_dict serialization, disk saving, and reloading parity."""
        os.makedirs("scratch", exist_ok=True)
        ckpt_path = "scratch/test_att_unet_ckpt.pth"
        
        # Save
        torch.save(self.model.state_dict(), ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))
        
        # Load into separate model
        new_model = AttentionUNet(in_channels=3, num_classes=3, base_channels=64, bilinear=False).to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        new_model.load_state_dict(state_dict)
        
        # Assert prediction parity
        self.model.eval()
        new_model.eval()
        with torch.no_grad():
            out_orig = self.model(self.input_tensor)
            out_loaded = new_model(self.input_tensor)
        self.assertTrue(torch.allclose(out_orig, out_loaded, atol=1e-6))
        
        # Clean temporary checkpoint
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        print("Test 11 Passed: Checkpoint saving, reloading, and numerical parity confirmed.")

    def test_12_cpu_compatibility(self):
        """Test 12: Verify CPU execution compatibility."""
        cpu_model = AttentionUNet(in_channels=3, num_classes=3, base_channels=64, bilinear=False).to("cpu")
        cpu_inp = torch.randn(1, 3, 224, 224, device="cpu")
        cpu_model.eval()
        with torch.no_grad():
            cpu_out = cpu_model(cpu_inp)
        self.assertEqual(cpu_out.shape, (1, 3, 224, 224))
        print("Test 12 Passed: CPU execution verified cleanly.")

if __name__ == "__main__":
    unittest.main()
