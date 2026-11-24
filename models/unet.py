import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Up(nn.Module):
    """
    Upsampling block: ConvTranspose2d (or bilinear upsample) followed by DoubleConv with skip connection
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if feature map sizes do not match exactly
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final 1x1 convolution projecting intermediate channels to the target class logits
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """
    Canonical 2D U-Net architecture based on Ronneberger et al. (2015).
    
    Specifications:
    - 4 encoder stages with two 3x3 convs, BatchNorm, and ReLU.
    - 2x2 max-pooling downsampling.
    - Bottleneck stage (1,024 channels with base_channels=64).
    - Symmetric 4-stage decoder with 2x2 transposed convolutions and skip connections.
    - 1x1 convolution output projection.
    - No attention gates, no transformer layers, trained completely from scratch.
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

        b = base_channels  # e.g., 64 (canonical Ronneberger et al., 2015)
        
        # Encoder stages
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)        # 64 -> 128
        self.down2 = Down(b * 2, b * 4)    # 128 -> 256
        self.down3 = Down(b * 4, b * 8)    # 256 -> 512
        factor = 2 if bilinear else 1
        self.down4 = Down(b * 8, (b * 16) // factor)  # 512 -> 1024 (Bottleneck)

        # Decoder stages
        self.up1 = Up(b * 16, (b * 8) // factor, bilinear)  # 1024 -> 512
        self.up2 = Up(b * 8, (b * 4) // factor, bilinear)   # 512 -> 256
        self.up3 = Up(b * 4, (b * 2) // factor, bilinear)   # 256 -> 128
        self.up4 = Up(b * 2, b, bilinear)                   # 128 -> 64
        self.outc = OutConv(b, num_classes)                 # 64 -> num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def count_parameters(self) -> dict:
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
        decoder_params = (
            sum(p.numel() for p in self.up1.parameters()) +
            sum(p.numel() for p in self.up2.parameters()) +
            sum(p.numel() for p in self.up3.parameters()) +
            sum(p.numel() for p in self.up4.parameters())
        )
        head_params = sum(p.numel() for p in self.outc.parameters())
        
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "encoder_parameters": encoder_params,
            "bottleneck_parameters": bottleneck_params,
            "decoder_parameters": decoder_params,
            "head_parameters": head_params
        }
