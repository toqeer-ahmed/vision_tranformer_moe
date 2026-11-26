import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class DecoderBlock(nn.Module):
    """
    Standard U-Net decoder block:
    Upsample (bilinear or transposed conv) -> Concat Skip -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> ReLU
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_transpose_conv: bool = False
    ):
        super().__init__()
        self.use_transpose_conv = use_transpose_conv
        
        if use_transpose_conv:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Identity()
            
        combined_channels = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_transpose_conv:
            x = self.up(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            
        return self.conv(x)


class ResNet50UNet(nn.Module):
    """
    U-Net segmentation architecture with an ImageNet-1K pretrained ResNet-50 backbone.
    
    Encoder Feature Hierarchy (for 224x224 input):
      - Stage 0 (Stem):  e0 -> [B, 64, 112, 112] (conv1 + bn1 + relu)
      - Stage 1 (Layer1): e1 -> [B, 256, 56, 56]
      - Stage 2 (Layer2): e2 -> [B, 512, 28, 28]
      - Stage 3 (Layer3): e3 -> [B, 1024, 14, 14]
      - Stage 4 (Layer4): e4 -> [B, 2048, 7, 7]  (Bottleneck)
      
    Decoder Channels:
      - Block 1 (7x7 -> 14x14):    2048 + 1024 (skip e3) -> 256
      - Block 2 (14x14 -> 28x28):   256 + 512  (skip e2) -> 128
      - Block 3 (28x28 -> 56x56):   128 + 256  (skip e1) -> 64
      - Block 4 (56x56 -> 112x112):  64 + 64   (skip e0) -> 32
      - Block 5 (112x112 -> 224x224): 32 + 0   (no skip) -> 16
      - Segmentation Head: 16 -> num_classes (1x1 conv)
      
    Parameter Budget:
      - Pretrained ResNet-50 Encoder: 23,508,032 parameters
      - U-Net Decoder & Head:          9,012,945 parameters (num_classes=1)
      - Total Trainable Parameters:   32,520,977 parameters (~32.52M)
    """
    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        weights: Optional[Union[str, ResNet50_Weights]] = "DEFAULT",
        freeze_encoder: bool = False,
        freeze_bn: bool = False,
        use_transpose_conv: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder
        self.freeze_bn = freeze_bn
        
        # 1. Initialize ResNet-50 Encoder
        if pretrained:
            if weights == "DEFAULT" or weights is None or weights == "IMAGENET1K_V1":
                weight_enum = ResNet50_Weights.IMAGENET1K_V1
            elif weights == "IMAGENET1K_V2":
                weight_enum = ResNet50_Weights.IMAGENET1K_V2
            elif isinstance(weights, ResNet50_Weights):
                weight_enum = weights
            elif isinstance(weights, str) and os.path.isfile(weights):
                weight_enum = None
            else:
                raise ValueError(f"Unsupported pretrained weights specification: {weights}")
                
            try:
                if weight_enum is not None:
                    resnet = models.resnet50(weights=weight_enum)
                else:
                    resnet = models.resnet50(weights=None)
                    state = torch.load(weights, map_location="cpu")
                    resnet.load_state_dict(state)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load ImageNet-1K pretrained weights for ResNet-50. "
                    f"Pretraining is strictly required by the research protocol. Error: {e}"
                )
        else:
            resnet = models.resnet50(weights=None)
            
        # Encoder stages
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # Output: 256
        self.layer2 = resnet.layer2  # Output: 512
        self.layer3 = resnet.layer3  # Output: 1024
        self.layer4 = resnet.layer4  # Output: 2048
        
        # 2. Initialize Decoder Stages
        self.decoder_block1 = DecoderBlock(2048, 1024, 256, use_transpose_conv=use_transpose_conv)
        self.decoder_block2 = DecoderBlock(256, 512, 128, use_transpose_conv=use_transpose_conv)
        self.decoder_block3 = DecoderBlock(128, 256, 64, use_transpose_conv=use_transpose_conv)
        self.decoder_block4 = DecoderBlock(64, 64, 32, use_transpose_conv=use_transpose_conv)
        self.decoder_block5 = DecoderBlock(32, 0, 16, use_transpose_conv=use_transpose_conv)
        
        # 3. Segmentation Head (1x1 Conv)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)
        
        # Apply pretraining controls
        if freeze_encoder:
            self._freeze_encoder_layers()
        if freeze_bn:
            self._freeze_encoder_bn()

    def _freeze_encoder_layers(self):
        """Freezes all parameters in the ResNet-50 encoder."""
        for module in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = False

    def _freeze_encoder_bn(self):
        """Sets BatchNorm in encoder to eval mode and freezes running stats."""
        for module in [self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        
        # Encoder forward pass
        e0 = self.relu(self.bn1(self.conv1(x)))      # [B, 64, H/2, W/2] = [B, 64, 112, 112]
        p = self.maxpool(e0)                         # [B, 64, H/4, W/4] = [B, 64, 56, 56]
        
        e1 = self.layer1(p)                          # [B, 256, H/4, W/4] = [B, 256, 56, 56]
        e2 = self.layer2(e1)                         # [B, 512, H/8, W/8] = [B, 512, 28, 28]
        e3 = self.layer3(e2)                         # [B, 1024, H/16, W/16] = [B, 1024, 14, 14]
        e4 = self.layer4(e3)                         # [B, 2048, H/32, W/32] = [B, 2048, 7, 7]
        
        # Decoder forward pass with skip connections
        d1 = self.decoder_block1(e4, e3)             # [B, 256, 14, 14]
        d2 = self.decoder_block2(d1, e2)             # [B, 128, 28, 28]
        d3 = self.decoder_block3(d2, e1)             # [B, 64, 56, 56]
        d4 = self.decoder_block4(d3, e0)             # [B, 32, 112, 112]
        d5 = self.decoder_block5(d4, None)           # [B, 16, 224, 224]
        
        logits = self.head(d5)                       # [B, num_classes, 224, 224]
        
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=True)
            
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """
        Calculates exact parameter counts programmatically across components.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        
        encoder_modules = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        encoder_params = sum(p.numel() for mod in encoder_modules for p in mod.parameters())
        
        decoder_modules = [
            self.decoder_block1, self.decoder_block2, self.decoder_block3,
            self.decoder_block4, self.decoder_block5
        ]
        decoder_params = sum(p.numel() for mod in decoder_modules for p in mod.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "encoder_parameters": encoder_params,
            "decoder_parameters": decoder_params,
            "head_parameters": head_params
        }
