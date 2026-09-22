import torch
import torch.nn as nn

class MLPExpert(nn.Module):
    """
    Standard Feed-Forward Expert block.
    Processes tokens dispatched to this expert.
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Dispatched tokens tensor of shape (num_dispatched_tokens, hidden_dim)
            
        Returns:
            torch.Tensor: Processed tokens tensor of shape (num_dispatched_tokens, hidden_dim)
        """
        return self.dropout(self.dense2(self.act(self.dense1(x))))

class SimpleMLPBlock(nn.Module):
    """
    Drop-in replacement for SegformerMixMLP without the 3x3 depthwise convolution.
    Used for Experiment B ablation baseline.
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, height: int = None, width: int = None) -> torch.Tensor:
        x = self.dense1(hidden_states)
        x = self.act(x)
        x = self.dense2(x)
        return self.dropout(x)

class SpatiallyAwareExpert(nn.Module):
    """
    Expert that preserves the 3x3 depthwise convolution of MixFFN.
    Requires Dense Routing (receives full image tensor) to compute correctly.
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, intermediate_dim)
        self.dwconv = nn.Conv2d(intermediate_dim, intermediate_dim, 3, 1, 1, groups=intermediate_dim)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        B, seq_len, _ = hidden_states.shape
        x = self.dense1(hidden_states)
        
        x = x.transpose(1, 2).contiguous().view(B, -1, height, width)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        
        x = self.act(x)
        x = self.dense2(x)
        return self.dropout(x)
