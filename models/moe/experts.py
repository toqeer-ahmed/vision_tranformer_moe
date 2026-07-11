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
