import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    """
    Gating network to compute routing logits for experts.
    Supports standard linear gating with optional training-time noise to encourage exploration.
    """
    def __init__(self, hidden_dim: int, num_experts: int, noisy_gating: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        
        self.wg = nn.Linear(hidden_dim, num_experts, bias=False)
        if noisy_gating:
            self.w_noise = nn.Linear(hidden_dim, num_experts, bias=False)
            self.softplus = nn.Softplus()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Token representations of shape (num_tokens, hidden_dim)
            
        Returns:
            torch.Tensor: Routing logits of shape (num_tokens, num_experts)
        """
        logits = self.wg(x)
        
        if self.noisy_gating and self.training:
            # Scale Gaussian noise by the softplus activation of noise logits
            noise_logits = self.w_noise(x)
            noise_scale = self.softplus(noise_logits)
            noise = torch.randn_like(logits) * noise_scale
            logits = logits + noise
            
        return logits
