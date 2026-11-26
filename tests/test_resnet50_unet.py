import unittest
import torch
import torch.nn as nn
import os
import sys
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# Ensure root directory is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.resnet50_unet import ResNet50UNet, DecoderBlock
from evaluation.losses import CombinedSegmentationLoss
from evaluation.metrics import SegmentationMetricAccumulator

class TestResNet50UNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Test Setup] Running ResNet-50 U-Net test suite on device: {cls.device}")

    def setUp(self):
        self.device = self.__class__.device

    def test_01_pretrained_weights_loading_and_verification(self):
        """Test 1: Verify ImageNet-1K pretrained weights are strictly loaded and match official weights."""
        model = ResNet50UNet(num_classes=1, pretrained=True, weights="IMAGENET1K_V1").to(self.device)
        self.assertTrue(model.pretrained)
        
        # Load official reference torchvision resnet50 weights
        ref_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Verify conv1 weight matches exactly
        self.assertTrue(
            torch.allclose(model.conv1.weight.data.cpu(), ref_resnet.conv1.weight.data, atol=1e-7),
            "ResNet-50 encoder conv1 weights do not match official ImageNet-1K V1 weights!"
        )
        # Verify layer4 weight matches exactly
        self.assertTrue(
            torch.allclose(model.layer4[0].conv1.weight.data.cpu(), ref_resnet.layer4[0].conv1.weight.data, atol=1e-7),
            "ResNet-50 encoder layer4 weights do not match official ImageNet-1K V1 weights!"
        )
        
        # Verify error is raised if invalid weights specification is passed
        with self.assertRaises((ValueError, RuntimeError)):
            _ = ResNet50UNet(pretrained=True, weights="NON_EXISTENT_WEIGHTS_FILE.pth")
            
        print("Test 1 Passed: Official ImageNet-1K V1 weights verified with strict integrity check.")

    def test_02_encoder_feature_hierarchy_dimensions(self):
        """Test 2: Verify feature resolutions and channel dimensions through all 5 encoder stages."""
        model = ResNet50UNet(num_classes=1, pretrained=False).to(self.device)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224, device=self.device)
        with torch.no_grad():
            e0 = model.relu(model.bn1(model.conv1(x)))
            p = model.maxpool(e0)
            e1 = model.layer1(p)
            e2 = model.layer2(e1)
            e3 = model.layer3(e2)
            e4 = model.layer4(e3)
            
        self.assertEqual(e0.shape, (2, 64, 112, 112), f"Stage 0 (Stem) shape mismatch: {e0.shape}")
        self.assertEqual(p.shape, (2, 64, 56, 56), f"Stem maxpool shape mismatch: {p.shape}")
        self.assertEqual(e1.shape, (2, 256, 56, 56), f"Stage 1 (Layer1) shape mismatch: {e1.shape}")
        self.assertEqual(e2.shape, (2, 512, 28, 28), f"Stage 2 (Layer2) shape mismatch: {e2.shape}")
        self.assertEqual(e3.shape, (2, 1024, 14, 14), f"Stage 3 (Layer3) shape mismatch: {e3.shape}")
        self.assertEqual(e4.shape, (2, 2048, 7, 7), f"Stage 4 (Layer4 / Bottleneck) shape mismatch: {e4.shape}")
        print("Test 2 Passed: Encoder feature hierarchy verified (64@112 -> 256@56 -> 512@28 -> 1024@14 -> 2048@7).")

    def test_03_decoder_and_output_shapes(self):
        """Test 3: Verify single-channel binary (num_classes=1) and 3-class output shapes (B, C, 224, 224)."""
        # Test binary output (num_classes=1)
        model_binary = ResNet50UNet(num_classes=1, pretrained=False).to(self.device)
        model_binary.eval()
        x = torch.randn(2, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out_bin = model_binary(x)
        self.assertEqual(out_bin.shape, (2, 1, 224, 224), f"Binary output shape mismatch: {out_bin.shape}")
        self.assertFalse(torch.isnan(out_bin).any(), "Binary output contains NaNs")

        # Test multi-class output (num_classes=3)
        model_multi = ResNet50UNet(num_classes=3, pretrained=False).to(self.device)
        model_multi.eval()
        with torch.no_grad():
            out_multi = model_multi(x)
        self.assertEqual(out_multi.shape, (2, 3, 224, 224), f"Multi-class output shape mismatch: {out_multi.shape}")
        self.assertFalse(torch.isnan(out_multi).any(), "Multi-class output contains NaNs")
        print("Test 3 Passed: Output shapes match (B, 1, 224, 224) and (B, 3, 224, 224).")

    def test_04_batch_size_scaling(self):
        """Test 4: Verify stability across variable batch sizes (B=1, B=2, B=4, B=8)."""
        model = ResNet50UNet(num_classes=1, pretrained=False).to(self.device)
        model.eval()
        with torch.no_grad():
            for b in [1, 2, 4, 8]:
                inp = torch.randn(b, 3, 224, 224, device=self.device)
                out = model(inp)
                self.assertEqual(out.shape, (b, 1, 224, 224), f"Failure at batch size {b}")
                self.assertFalse(torch.isnan(out).any())
        print("Test 4 Passed: Scalability verified across batch sizes 1, 2, 4, 8.")

    def test_05_loss_and_backpropagation(self):
        """Test 5: Verify Focal Tversky Loss and backpropagation gradient flow through entire network."""
        model = ResNet50UNet(num_classes=3, pretrained=False).to(self.device)
        model.train()
        criterion = CombinedSegmentationLoss(loss_type="focal_tversky", dice_weight=1.0).to(self.device)
        
        images = torch.randn(2, 3, 224, 224, device=self.device)
        targets = torch.randint(0, 3, (2, 224, 224), device=self.device).long()
        
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        self.assertFalse(torch.isnan(loss), "Loss computed as NaN")
        self.assertGreater(loss.item(), 0.0, "Loss must be positive")
        
        loss.backward()
        
        # Verify gradients on encoder stem conv1, layer4 bottleneck, and decoder head
        self.assertIsNotNone(model.conv1.weight.grad, "conv1 has None gradient")
        self.assertGreater(model.conv1.weight.grad.norm().item(), 0.0, "conv1 received 0 gradient")
        
        self.assertIsNotNone(model.layer4[0].conv1.weight.grad, "layer4 has None gradient")
        self.assertGreater(model.layer4[0].conv1.weight.grad.norm().item(), 0.0, "layer4 received 0 gradient")
        
        self.assertIsNotNone(model.decoder_block1.conv[0].weight.grad, "decoder_block1 has None gradient")
        self.assertGreater(model.decoder_block1.conv[0].weight.grad.norm().item(), 0.0, "decoder_block1 received 0 gradient")
        
        self.assertIsNotNone(model.head.weight.grad, "head has None gradient")
        self.assertGreater(model.head.weight.grad.norm().item(), 0.0, "head received 0 gradient")
        print(f"Test 5 Passed: Full backprop verified from head through decoder and encoder stem (loss = {loss.item():.4f}).")

    def test_06_exact_parameter_counts(self):
        """Test 6: Verify exact parameter count assertions for both binary and 3-class configurations."""
        # Binary configuration (num_classes=1)
        model_bin = ResNet50UNet(num_classes=1, pretrained=False)
        counts_bin = model_bin.count_parameters()
        
        expected_encoder = 23_508_032
        expected_decoder = 9_012_928
        expected_head_bin = 17       # 16*1 + 1 bias
        expected_total_bin = 32_520_977  # 23,508,032 + 9,012,928 + 17
        
        self.assertEqual(counts_bin["encoder_parameters"], expected_encoder, "Encoder parameter count mismatch")
        self.assertEqual(counts_bin["decoder_parameters"], expected_decoder, "Decoder parameter count mismatch")
        self.assertEqual(counts_bin["head_parameters"], expected_head_bin, "Head parameter count mismatch for num_classes=1")
        self.assertEqual(counts_bin["total_parameters"], expected_total_bin, f"Total parameters must be exactly {expected_total_bin:,}")
        self.assertEqual(counts_bin["trainable_parameters"], expected_total_bin, "All parameters must be trainable by default")
        
        # 3-Class configuration (num_classes=3)
        model_multi = ResNet50UNet(num_classes=3, pretrained=False)
        counts_multi = model_multi.count_parameters()
        expected_head_multi = 51     # 16*3 + 3 bias
        expected_total_multi = 32_521_011
        
        self.assertEqual(counts_multi["head_parameters"], expected_head_multi, "Head parameter count mismatch for num_classes=3")
        self.assertEqual(counts_multi["total_parameters"], expected_total_multi, f"Total parameters must be exactly {expected_total_multi:,}")
        
        print("\n=== RESNET-50 U-NET PARAMETER AUDIT ===")
        print(f"Encoder Parameters:        {counts_bin['encoder_parameters']:,} (23.51M)")
        print(f"Decoder Blocks (1-5):      {counts_bin['decoder_parameters']:,} (9.01M)")
        print(f"Head Parameters (C=1):     {counts_bin['head_parameters']:,}")
        print(f"Head Parameters (C=3):     {counts_multi['head_parameters']:,}")
        print(f"Total Parameters (C=1):    {counts_bin['total_parameters']:,} (~32.52M)")
        print(f"Total Parameters (C=3):    {counts_multi['total_parameters']:,} (~32.52M)")
        print(f"Comparison with Swin-Tiny: 29.29M (~1.11x)")
        print(f"Comparison with U-Net:     31.04M (~1.05x)")
        print("Test 6 Passed: Exact parameter count assertions verified programmatically.")

    def test_07_freeze_encoder_control(self):
        """Test 7: Verify freeze_encoder=True isolates encoder parameters while keeping decoder trainable."""
        model_frozen = ResNet50UNet(num_classes=1, pretrained=False, freeze_encoder=True).to(self.device)
        counts = model_frozen.count_parameters()
        
        expected_trainable = counts["decoder_parameters"] + counts["head_parameters"]
        expected_non_trainable = counts["encoder_parameters"]
        
        self.assertEqual(counts["trainable_parameters"], expected_trainable)
        self.assertEqual(counts["non_trainable_parameters"], expected_non_trainable)
        
        # Verify grad requirement on individual tensors
        for param in model_frozen.conv1.parameters():
            self.assertFalse(param.requires_grad)
        for param in model_frozen.layer4.parameters():
            self.assertFalse(param.requires_grad)
        for param in model_frozen.decoder_block1.parameters():
            self.assertTrue(param.requires_grad)
        for param in model_frozen.head.parameters():
            self.assertTrue(param.requires_grad)
            
        print("Test 7 Passed: Freeze encoder mechanism verified.")

    def test_08_checkpoint_save_and_reload(self):
        """Test 8: Verify checkpoint serialization and strict numerical parity upon reload."""
        os.makedirs("scratch", exist_ok=True)
        ckpt_path = "scratch/test_resnet50_unet_ckpt.pth"
        
        model = ResNet50UNet(num_classes=3, pretrained=False).to(self.device)
        x = torch.randn(2, 3, 224, 224, device=self.device)
        
        model.eval()
        with torch.no_grad():
            orig_output = model(x)
            
        torch.save(model.state_dict(), ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path), "Checkpoint file was not written")
        
        reloaded_model = ResNet50UNet(num_classes=3, pretrained=False).to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        reloaded_model.load_state_dict(state_dict)
        reloaded_model.eval()
        
        with torch.no_grad():
            reloaded_output = reloaded_model(x)
            
        self.assertTrue(
            torch.allclose(orig_output, reloaded_output, atol=1e-6),
            "Reloaded model outputs do not match original model outputs!"
        )
        
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        print("Test 8 Passed: Checkpoint serialization and numerical reload parity verified.")

    def test_09_metric_accumulator_compatibility(self):
        """Test 9: Verify SegmentationMetricAccumulator integrates with ResNet-50 U-Net predictions."""
        acc = SegmentationMetricAccumulator(num_classes=3)
        model = ResNet50UNet(num_classes=3, pretrained=False).to(self.device)
        model.eval()
        
        x = torch.randn(4, 3, 224, 224, device=self.device)
        target = torch.randint(0, 3, (4, 224, 224), device=self.device).long()
        
        with torch.no_grad():
            logits = model(x)
            preds = logits.argmax(dim=1)
            
        acc.update(preds, target)
        metrics = acc.compute()
        
        for k in ["pixel_accuracy", "mean_iou", "mean_dice", "mean_precision", "mean_recall"]:
            self.assertIn(k, metrics, f"Missing metric key: {k}")
            self.assertTrue(0.0 <= metrics[k] <= 1.0, f"Metric {k} out of bounds: {metrics[k]}")
            
        print("Test 9 Passed: Metric accumulator successfully computed valid metrics.")

if __name__ == "__main__":
    unittest.main()
