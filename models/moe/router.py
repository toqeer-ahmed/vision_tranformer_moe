import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """
    Router that assigns tokens to top-k experts.
    Also computes the auxiliary load balancing loss to prevent representation collapse/expert starvation.
    """
    def __init__(self, num_experts: int, top_k: int = 1, balance_loss_coef: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coef = balance_loss_coef
        
    def forward(self, gating_logits: torch.Tensor):
        """
        Forward pass.
        
        Args:
            gating_logits (torch.Tensor): Logits of shape (num_tokens, num_experts)
            
        Returns:
            tuple: (top_k_gates, top_k_indices, balance_loss)
        """
        num_tokens = gating_logits.size(0)
        
        # Apply Softmax to get routing probability distribution
        gates = F.softmax(gating_logits, dim=-1)
        
        # Get Top-k gating values and indices
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        
        # Normalize weights over selection
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Calculate auxiliary loss (load balancing)
        if self.training:
            # Fraction of tokens dispatched to expert i (over top-k choices)
            assignments = torch.zeros(num_tokens, self.num_experts, device=gating_logits.device)
            assignments.scatter_(1, top_k_indices, 1.0)
            f = assignments.mean(dim=0) # shape: (num_experts)
            
            # Average gate probability assigned to expert i
            P = gates.mean(dim=0) # shape: (num_experts)
            
            # Auxiliary Load balancing loss formula: N * sum(f_i * P_i)
            balance_loss = self.num_experts * torch.sum(f * P)
        else:
            balance_loss = torch.tensor(0.0, device=gating_logits.device)
            
        return top_k_gates, top_k_indices, balance_loss * self.balance_loss_coef
