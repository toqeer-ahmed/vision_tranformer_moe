import torch
import torch.nn as nn

try:
    from .experts import MLPExpert
    from .gating import GatingNetwork
    from .router import TopKRouter
except ImportError:
    from experts import MLPExpert
    from gating import GatingNetwork
    from router import TopKRouter

class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) layer wrapping GatingNetwork, TopKRouter, and MLPExpert modules.
    Conforms to the SegformerMixMLP signature to allow drop-in replacement.
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
        
        # Expert hidden dimension size
        intermediate_dim = 4 * hidden_dim
        
        # Initialize experts ModuleList
        self.experts = nn.ModuleList([
            MLPExpert(hidden_dim, intermediate_dim) for _ in range(num_experts)
        ])
        
        # Initialize gating network and top-k router
        self.gating = GatingNetwork(hidden_dim, num_experts, noisy_gating)
        self.router = TopKRouter(num_experts, top_k, balance_loss_coef)
        
        # Store auxiliary load-balancing loss for optimization access
        self.register_buffer("aux_loss", torch.tensor(0.0))
        
    def forward(self, hidden_states: torch.Tensor, height: int = None, width: int = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states (torch.Tensor): shape (batch, seq_len, hidden_dim)
            height (int, optional): Height of spatial grid (conforms to SegformerMixMLP)
            width (int, optional): Width of spatial grid (conforms to SegformerMixMLP)
            
        Returns:
            torch.Tensor: output tensor of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Reshape hidden_states into 2D tensor of tokens
        tokens = hidden_states.view(-1, hidden_dim) # shape: (batch_size * seq_len, hidden_dim)
        
        # Compute routing gating logits
        gating_logits = self.gating(tokens)
        
        # Determine routing assignment and compute balance loss
        top_k_gates, top_k_indices, aux_loss = self.router(gating_logits)
        self.aux_loss = aux_loss
        self.last_routing_indices = top_k_indices.detach().cpu()
        
        # Initialize token outputs
        output = torch.zeros_like(tokens)
        
        # Execute expert computation for assigned tokens
        for exp_idx, expert in enumerate(self.experts):
            # Check which tokens route to this expert
            mask = (top_k_indices == exp_idx) # shape: (num_tokens, top_k)
            token_mask = mask.any(dim=-1) # shape: (num_tokens)
            
            if token_mask.any():
                # Extract subset of tokens dispatched to expert
                dispatched_tokens = tokens[token_mask]
                expert_outputs = expert(dispatched_tokens)
                
                # Fetch gate weights for this expert
                row_indices, col_indices = torch.where(mask)
                gate_weights = torch.zeros(tokens.size(0), device=tokens.device)
                gate_weights[row_indices] = top_k_gates[row_indices, col_indices]
                
                # Weight outputs and accumulate into unified token representation
                weighted_outputs = expert_outputs * gate_weights[token_mask].unsqueeze(-1)
                output[token_mask] += weighted_outputs
                
        # Reshape back to original 3D shape
        return output.view(batch_size, seq_len, hidden_dim)
