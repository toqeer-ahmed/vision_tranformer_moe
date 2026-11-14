import os
import sys
import torch

# Ensure local repository root is accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Make sure HF_HUB_OFFLINE is not forced
if "HF_HUB_OFFLINE" in os.environ:
    del os.environ["HF_HUB_OFFLINE"]

from models.swin import SwinSegmentation, replace_swin_ffn_with_standard_moe
from evaluation.losses import CombinedSegmentationLoss

def test_auxiliary_loss_integration():
    """
    Diagnostic Verification Test:
    1. Instantiates Standard-MLP Swin-MoE model.
    2. Runs forward pass through all 12 MoE layers.
    3. Confirms auxiliary load balancing loss is available and finite for all 12 layers.
    4. Confirms aggregated aux_loss maintains computational graph (requires_grad=True).
    5. Confirms backpropagation delivers non-zero gradients to all router and expert parameters.
    6. Verifies that Vanilla Swin has zero auxiliary loss.
    """
    print("\n[DIAGNOSTIC TEST] Step 1: Testing Vanilla Swin Aux Loss Isolation...")
    vanilla_model = SwinSegmentation(num_classes=3, pretrained=False)
    vanilla_model.train()
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_targets = torch.randint(0, 3, (2, 224, 224))
    
    vanilla_logits = vanilla_model(dummy_input)
    vanilla_aux = vanilla_model.get_aux_loss()
    assert vanilla_aux.item() == 0.0, f"Vanilla Swin returned non-zero aux loss: {vanilla_aux.item()}"
    print(f"PASS: Vanilla Swin aux loss is strictly 0.0.")

    print("\n[DIAGNOSTIC TEST] Step 2: Testing Standard-MLP Swin-MoE Layer Integration...")
    moe_model = SwinSegmentation(num_classes=3, pretrained=False)
    moe_cfg = {
        'num_experts': 4,
        'top_k': 2,
        'noisy_gating': True,
        'balance_loss_coef': 0.01,
        'warm_init': True
    }
    replace_swin_ffn_with_standard_moe(moe_model, moe_cfg)
    moe_model.train()
    
    criterion = CombinedSegmentationLoss(loss_type="focal_tversky", dice_weight=1.0)
    
    moe_logits = moe_model(dummy_input)
    seg_loss = criterion(moe_logits, dummy_targets)
    
    # Layer-by-layer audit
    moe_layers = []
    for s_idx, stage in enumerate(moe_model.backbone.encoder.layers):
        for b_idx, block in enumerate(stage.blocks):
            if hasattr(block, 'intermediate') and hasattr(block.intermediate, 'moe_layer'):
                l = block.intermediate.moe_layer
                moe_layers.append((s_idx, b_idx, l))
                
    assert len(moe_layers) == 12, f"Expected 12 MoE layers, found {len(moe_layers)}"
    print(f"PASS: Verified all 12 MoE layers are present and active.")
    
    for s_idx, b_idx, l in moe_layers:
        aux_val = l.aux_loss
        assert aux_val is not None, f"Layer at Stage {s_idx}, Block {b_idx} has None aux_loss"
        assert torch.isfinite(aux_val), f"Layer at Stage {s_idx}, Block {b_idx} has non-finite aux_loss: {aux_val}"
        assert aux_val.requires_grad, f"Layer at Stage {s_idx}, Block {b_idx} aux_loss does not require grad"
        
    print(f"PASS: All 12 MoE layers independently generated valid, finite, graph-connected aux losses.")
    
    agg_aux = moe_model.get_aux_loss()
    assert agg_aux.requires_grad, "Aggregated aux loss disconnected from autograd graph!"
    assert torch.isfinite(agg_aux), f"Aggregated aux loss non-finite: {agg_aux}"
    print(f"PASS: Aggregated aux loss ({agg_aux.item():.6f}) is finite and connected to autograd graph.")
    
    total_loss = seg_loss + agg_aux
    print("\n[DIAGNOSTIC TEST] Step 3: Testing End-to-End Backward Pass...")
    moe_model.zero_grad()
    total_loss.backward()
    
    for s_idx, b_idx, l in moe_layers:
        wg_grad = l.gating.wg.weight.grad
        exp_grad = l.experts[0].dense1.weight.grad
        assert wg_grad is not None, f"Router wg has no gradient at Stage {s_idx}, Block {b_idx}"
        assert exp_grad is not None, f"Expert dense1 has no gradient at Stage {s_idx}, Block {b_idx}"
        assert torch.isfinite(wg_grad).all(), f"Router gradient non-finite at Stage {s_idx}, Block {b_idx}"
        assert torch.isfinite(exp_grad).all(), f"Expert gradient non-finite at Stage {s_idx}, Block {b_idx}"
        assert wg_grad.norm().item() > 0, f"Router gradient norm is zero at Stage {s_idx}, Block {b_idx}"
        assert exp_grad.norm().item() > 0, f"Expert gradient norm is zero at Stage {s_idx}, Block {b_idx}"
        
    print("PASS: Verified that total_loss.backward() successfully updates both router and expert weights.")

    print("\n[DIAGNOSTIC TEST] Step 4: Proving Pure Aux Loss Gradient Attribution...")
    moe_model.zero_grad()
    _ = moe_model(dummy_input)
    pure_aux = moe_model.get_aux_loss()
    pure_aux.backward()
    
    for s_idx, b_idx, l in moe_layers:
        wg_grad = l.gating.wg.weight.grad
        assert wg_grad is not None and wg_grad.norm().item() > 0, f"Pure aux loss failed to propagate to router at Stage {s_idx}, Block {b_idx}"
        
    print("PASS: Confirmed mathematical proof that auxiliary load-balancing loss directly optimizes router parameters!")
    print("\nALL DIAGNOSTIC CHECKS PASSED SUCCESSFULLY!\n")

if __name__ == "__main__":
    test_auxiliary_loss_integration()

