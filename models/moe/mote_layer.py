import torch
import torch.nn as nn
import torch.nn.functional as F

class MoTELayer(nn.Module):
    """
    Token-Level Mixture of Experts (MoTE) Routing.
    Takes N expert tokens, scores them using a linear gating network, 
    and selects the Top-K tokens based on confidence. Unselected tokens are zeroed out.
    """
    def __init__(self, token_dim=256, num_experts=4, top_k=2, noisy_gating=True):
        super(MoTELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noisy_gating = noisy_gating
        
        # Gating network to predict score from the token representation
        self.w_gate = nn.Parameter(torch.zeros(token_dim, 1))
        self.w_noise = nn.Parameter(torch.zeros(token_dim, 1))
        self.softplus = nn.Softplus()
        
        # Initialize gates slightly positively
        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)
        nn.init.normal_(self.w_noise, mean=0.0, std=0.02)

    def forward(self, expert_tokens: torch.Tensor):
        """
        Args:
            expert_tokens: [B, num_experts, token_dim]
        Returns:
            out_tokens: [B, num_experts, token_dim] (only top_k are non-zero)
            gating_probs: [B, num_experts] (for load balancing loss)
        """
        B, N, D = expert_tokens.shape
        assert N == self.num_experts, f"Expected {self.num_experts} expert tokens, got {N}"
        
        # Compute logits for each token: [B, N, 1]
        clean_logits = expert_tokens @ self.w_gate
        
        if self.noisy_gating and self.training:
            raw_noise = expert_tokens @ self.w_noise
            noise_std = self.softplus(raw_noise) + 1e-2
            epsilon = torch.randn_like(clean_logits)
            logits = clean_logits + noise_std * epsilon
        else:
            logits = clean_logits
            
        logits = logits.squeeze(-1) # [B, N]
        
        # Top-k routing selection
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        
        # Softmax over top-k to get confidence G(.)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Gating probabilities for load balancing over all experts
        gating_probs = F.softmax(logits, dim=-1) # [B, N]
        
        # Create output tensor with only top_k activated
        out_tokens = torch.zeros_like(expert_tokens)
        
        for b in range(B):
            for k in range(self.top_k):
                idx = top_k_indices[b, k]
                gate_val = top_k_gates[b, k]
                out_tokens[b, idx] = expert_tokens[b, idx] * gate_val
                
        return out_tokens, gating_probs
