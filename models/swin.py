import torch
import torch.nn as nn
from transformers import SwinModel, SwinConfig

class SwinSegmentation(nn.Module):
    def __init__(self, model_name="microsoft/swin-tiny-patch4-window7-224", num_classes=3, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = SwinModel.from_pretrained(model_name, add_pooling_layer=False)
        else:
            config = SwinConfig.from_pretrained(model_name)
            self.backbone = SwinModel(config, add_pooling_layer=False)
            
        self.hidden_size = self.backbone.config.hidden_size
        
        # Simple segmentation decode head
        self.decode_head = nn.Sequential(
            nn.Conv2d(self.hidden_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state # [B, L, C]
        
        B, L, C = sequence_output.shape
        H = W = int(L ** 0.5)
        
        x = sequence_output.transpose(1, 2).view(B, C, H, W)
        logits = self.decode_head(x)
        
        # Upsample back to image size (e.g., 224x224)
        logits = nn.functional.interpolate(
            logits, size=pixel_values.shape[-2:], mode='bilinear', align_corners=False
        )
        return logits

def replace_swin_ffn_with_moe(model, moe_cfg, logger=None):
    from models.moe.dense_moe_layer import DenseMoELayer
    
    replaced_count = 0
    
    class SwinMoEWrapper(nn.Module):
        def __init__(self, hidden_dim, moe_layer):
            super().__init__()
            self.moe_layer = moe_layer
        def forward(self, hidden_states):
            B, L, C = hidden_states.shape
            H = W = int(L ** 0.5)
            return self.moe_layer(hidden_states, H, W)

    for stage_idx, stage in enumerate(model.backbone.encoder.layers):
        for block_idx, block in enumerate(stage.blocks):
            hidden_dim = block.output.dense.in_features
            
            moe_layer = DenseMoELayer(
                hidden_dim=hidden_dim,
                num_experts=moe_cfg.get("num_experts", 4),
                top_k=moe_cfg.get("top_k", 2),
                noisy_gating=moe_cfg.get("noisy_gating", True),
                balance_loss_coef=moe_cfg.get("balance_loss_coef", 0.01)
            )
            
            # Replace intermediate with MoE (which handles the full expansion internally)
            block.intermediate = SwinMoEWrapper(hidden_dim, moe_layer)
            # Replace output with Identity because MoE already outputs the projected dimensions
            block.output = nn.Identity()
            replaced_count += 1
            
    return replaced_count
