import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

class DoubleConv(nn.Module):
    """
    Standard U-Net Double Convolution block:
    [Conv2d (3x3) -> BatchNorm2d -> ReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class AttentionGate(nn.Module):
    """
    Additive Attention Gate based on Oktay et al. (2018):
    alpha = Sigmoid(psi^T ReLU(W_g^T g + W_x^T x))
    x_hat = alpha * x
    
    Parameters:
    - F_g: Number of feature channels in gating signal g (from lower decoder stage)
    - F_l: Number of feature channels in skip connection x (from encoder stage)
    - F_int: Intermediate channel dimension for additive projection (typically F_l // 2)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        g: (B, F_g, H_g, W_g) - coarse gating signal from decoder
        x: (B, F_l, H_x, W_x) - fine skip connection from encoder
        returns: (x_hat, alpha) where x_hat is gated feature map and alpha is attention map
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # If spatial dimensions differ (e.g. padding/strides), interpolate gating signal
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
            
        f = self.relu(g1 + x1)
        alpha = self.psi(f)
        x_hat = x * alpha
        return x_hat, alpha

class Down(nn.Module):
    """
    Downsampling block: MaxPool2d (2x2) followed by DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class AttentionUp(nn.Module):
    """
    Upsampling block with Attention Gate:
    Upsample(scale_factor=2) -> AttentionGate(g, x_skip) -> Cat([x_hat, up_g]) -> DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.bilinear = bilinear
        skip_channels = out_channels
        gating_channels = in_channels
        inter_channels = max(skip_channels // 2, 1)
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            # When bilinear, upsampled features have in_channels
            self.ag = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=inter_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.ag = AttentionGate(F_g=in_channels // 2, F_l=skip_channels, F_int=inter_channels)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_prev: torch.Tensor, x_skip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.up(x_prev)
        
        # Match dimensions if odd padding
        diffY = x_skip.size(2) - g.size(2)
        diffX = x_skip.size(3) - g.size(3)
        if diffX != 0 or diffY != 0:
            g = F.pad(g, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x_hat, alpha = self.ag(g=g, x=x_skip)
        cat_features = torch.cat([x_hat, g], dim=1)
        out = self.conv(cat_features)
        return out, alpha

class OutConv(nn.Module):
    """
    Final 1x1 convolution projecting intermediate channels to the target class logits
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class AttentionUNet(nn.Module):
    """
    Canonical Attention U-Net architecture based on Oktay et al. (2018).
    
    Key Features:
    - 4-stage encoder with double 3x3 convs, BatchNorm, and ReLU.
    - 2x2 max-pooling downsampling.
    - Bottleneck stage (1,024 channels with base_channels=64).
    - Symmetric 4-stage decoder with ConvTranspose2d upsampling.
    - 4 Additive Attention Gates on skip connections.
    - 1x1 convolution output projection (num_classes).
    - Fully trainable from scratch (no pretrained weights).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 64,
        bilinear: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear

        b = base_channels  # Canonical Ronneberger & Oktay specification: 64

        # Encoder stages
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)        # 64 -> 128
        self.down2 = Down(b * 2, b * 4)    # 128 -> 256
        self.down3 = Down(b * 4, b * 8)    # 256 -> 512
        factor = 2 if bilinear else 1
        self.down4 = Down(b * 8, (b * 16) // factor)  # 512 -> 1024 (Bottleneck)

        # Decoder stages with Attention Gates
        self.up1 = AttentionUp(b * 16, (b * 8) // factor, bilinear)  # 1024 -> 512
        self.up2 = AttentionUp(b * 8, (b * 4) // factor, bilinear)   # 512 -> 256
        self.up3 = AttentionUp(b * 4, (b * 2) // factor, bilinear)   # 256 -> 128
        self.up4 = AttentionUp(b * 2, b, bilinear)                   # 128 -> 64
        self.outc = OutConv(b, num_classes)                          # 64 -> num_classes

        # Cache for last generated attention maps (for visualization & test inspection)
        self.last_attention_maps = []

    def forward(
        self,
        x: torch.Tensor,
        return_attentions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # Encoder forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder forward pass with attention gating
        d1, a1 = self.up1(x5, x4)
        d2, a2 = self.up2(d1, x3)
        d3, a3 = self.up3(d2, x2)
        d4, a4 = self.up4(d3, x1)
        
        logits = self.outc(d4)
        self.last_attention_maps = [a1, a2, a3, a4]

        if return_attentions:
            return logits, [a1, a2, a3, a4]
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """
        Calculates exact parameter counts programmatically across components.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        encoder_params = (
            sum(p.numel() for p in self.inc.parameters()) +
            sum(p.numel() for p in self.down1.parameters()) +
            sum(p.numel() for p in self.down2.parameters()) +
            sum(p.numel() for p in self.down3.parameters())
        )
        bottleneck_params = sum(p.numel() for p in self.down4.parameters())
        
        # Attention Gate parameters specifically
        ag_params = (
            sum(p.numel() for p in self.up1.ag.parameters()) +
            sum(p.numel() for p in self.up2.ag.parameters()) +
            sum(p.numel() for p in self.up3.ag.parameters()) +
            sum(p.numel() for p in self.up4.ag.parameters())
        )
        
        # Decoder convs and upsamplers (excluding AGs)
        decoder_conv_params = (
            sum(p.numel() for p in self.up1.up.parameters()) +
            sum(p.numel() for p in self.up1.conv.parameters()) +
            sum(p.numel() for p in self.up2.up.parameters()) +
            sum(p.numel() for p in self.up2.conv.parameters()) +
            sum(p.numel() for p in self.up3.up.parameters()) +
            sum(p.numel() for p in self.up3.conv.parameters()) +
            sum(p.numel() for p in self.up4.up.parameters()) +
            sum(p.numel() for p in self.up4.conv.parameters())
        )
        
        decoder_total_params = ag_params + decoder_conv_params
        head_params = sum(p.numel() for p in self.outc.parameters())

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "encoder_parameters": encoder_params,
            "bottleneck_parameters": bottleneck_params,
            "decoder_parameters": decoder_total_params,
            "decoder_conv_parameters": decoder_conv_params,
            "attention_gate_parameters": ag_params,
            "head_parameters": head_params
        }
