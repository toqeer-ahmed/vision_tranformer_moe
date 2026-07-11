import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig

class SegFormerSegmentation(nn.Module):
    """
    SegFormer wrapper for semantic segmentation.
    Wraps Hugging Face's SegformerForSemanticSegmentation (e.g. nvidia/mit-b0).
    """
    def __init__(
        self,
        model_name: str = "nvidia/mit-b0",
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        if pretrained:
            # Load pretrained weights, ignoring size mismatch for custom classes heads
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Initialize from configuration with random weights
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Interpolated segmentation logits of shape (batch_size, num_classes, height, width)
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits # shape: (batch_size, num_classes, H_out, W_out) where H_out/W_out is H/4, W/4
        
        # Bilinear interpolation back to input image size
        interpolated_logits = nn.functional.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        return interpolated_logits
