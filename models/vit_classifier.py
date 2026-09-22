import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class ViTClassifier(nn.Module):
    """
    Vision Transformer wrapper for image classification.
    Wraps Hugging Face's ViTForImageClassification.
    """
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 10,
        pretrained: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        if pretrained:
            # Load pretrained weights, ignore mismatched sizes to allow replacing the classification head
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Initialize from configuration with random weights
            config = ViTConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.vit = ViTForImageClassification(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        outputs = self.vit(pixel_values=x)
        return outputs.logits
