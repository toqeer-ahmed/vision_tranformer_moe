import torch
import torch.nn as nn

class ProgressivePromptTokenization(nn.Module):
    """
    Progressive Prompt Tokenization (PPT) module.
    Automatically generates sparse prompt tokens (replacing human clicks/boxes) 
    by cross-attending learnable queries with the extracted image features.
    """
    def __init__(self, embed_dim=256, num_queries=2, num_heads=8):
        super(ProgressivePromptTokenization, self).__init__()
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        
        # Multi-head Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # MLP Projection
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeddings: Output from SAM Encoder [B, 256, H, W]
        Returns:
            prompt_tokens: Generated prompt tokens [B, num_queries, 256]
        """
        B, C, H, W = image_embeddings.shape
        
        # Flatten spatial dimensions: [B, H*W, C]
        img_features = image_embeddings.view(B, C, -1).permute(0, 2, 1)
        
        # Expand queries for the batch: [B, num_queries, C]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Cross Attention: Q = queries, K = V = image features
        attn_out, _ = self.cross_attn(queries, img_features, img_features)
        out1 = self.norm1(queries + attn_out)
        
        # MLP and residual connection
        mlp_out = self.mlp(out1)
        prompt_tokens = self.norm2(out1 + mlp_out)
        
        return prompt_tokens
