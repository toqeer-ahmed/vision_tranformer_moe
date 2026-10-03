import torch
import torch.nn as nn

try:
    from .experts import SimpleMLPBlock
    from .gating import GatingNetwork
    from .router import TopKRouter
except ImportError:
    from experts import SimpleMLPBlock
    from gating import GatingNetwork
    from router import TopKRouter

class StandardSwinMoELayer(nn.Module):
    """
    Standard Mixture of Experts (MoE) layer for Swin Transformer (Experiment 9).
    Uses SimpleMLPBlock (Linear -> GELU -> Linear) as experts to exactly match Swin's native FFN.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        capacity_factor: float = 1.2,
        noisy_gating: bool = True,
        balance_loss_coef: float = 0.01
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Swin Transformer FFN expansion ratio is typically 4
        intermediate_dim = 4 * hidden_dim
        
        # Use SimpleMLPBlock instead of SpatiallyAwareExpert
        self.experts = nn.ModuleList([
            SimpleMLPBlock(hidden_dim, intermediate_dim) for _ in range(num_experts)
        ])
        
        self.gating = GatingNetwork(hidden_dim, num_experts, noisy_gating)
        self.router = TopKRouter(num_experts, top_k, balance_loss_coef)
        self.register_buffer("aux_loss", torch.tensor(0.0))
        
    def forward(self, hidden_states: torch.Tensor, height: int = None, width: int = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        tokens = hidden_states.view(-1, hidden_dim)
        
        gating_logits = self.gating(tokens)
        top_k_gates, top_k_indices, aux_loss = self.router(gating_logits)
        self.aux_loss = aux_loss
        self.last_routing_indices = top_k_indices.detach().cpu()
        
        output = torch.zeros_like(hidden_states)
        
        for exp_idx, expert in enumerate(self.experts):
            mask = (top_k_indices == exp_idx)
            token_mask = mask.any(dim=-1).view(batch_size, seq_len)
            
            if token_mask.any():
                # Process the entire grid using SimpleMLPBlock (dense computation)
                expert_outputs = expert(hidden_states, height, width) # [B, seq_len, hidden_dim]
                
                # Create mask tensor of shape [batch_size * seq_len]
                gate_weights = torch.zeros(tokens.size(0), device=tokens.device)
                row_indices, col_indices = torch.where(mask)
                gate_weights[row_indices] = top_k_gates[row_indices, col_indices]
                
                # Reshape to [B, seq_len, 1] to multiply with expert output
                gate_weights = gate_weights.view(batch_size, seq_len, 1)
                
                output += expert_outputs * gate_weights
                
        return output
