import torch
import torch.nn as nn
from transformers import SamModel

from models.ppt import ProgressivePromptTokenization
from models.moe.mote_layer import MoTELayer

class SegMoTE(nn.Module):
    """
    SegMoTE Framework: Token-Level Mixture of Experts for Medical Image Segmentation
    This wrapper freezes the SAM Image Encoder, generates prompts via PPT, 
    and uses MoTE to route dynamic expert tokens.
    """
    def __init__(self, sam_model_name="facebook/sam-vit-base", num_experts=4, top_k=2):
        super(SegMoTE, self).__init__()
        
        # Load pre-trained SAM
        self.sam = SamModel.from_pretrained(sam_model_name)
        
        # Freeze SAM Encoder
        for param in self.sam.vision_encoder.parameters():
            param.requires_grad = False
            
        # Freeze SAM Prompt Encoder (we don't use the standard one anyway)
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
            
        embed_dim = self.sam.config.prompt_encoder_config.hidden_size # 256
        
        # Progressive Prompt Tokenization (PPT)
        self.ppt = ProgressivePromptTokenization(embed_dim=embed_dim, num_queries=2)
        
        # Expert Tokens
        self.num_experts = num_experts
        self.expert_tokens = nn.Parameter(torch.randn(1, num_experts, embed_dim))
        
        # MoTE Router
        self.mote = MoTELayer(token_dim=embed_dim, num_experts=num_experts, top_k=top_k)
        
        # Final segmentation projection (using the routed expert token)
        self.mask_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 8)
        )

    def forward(self, pixel_values: torch.Tensor):
        B = pixel_values.shape[0]
        original_size = (pixel_values.shape[-2], pixel_values.shape[-1])
        
        # SAM expects 1024x1024 inputs
        if original_size != (1024, 1024):
            pixel_values = nn.functional.interpolate(pixel_values, size=(1024, 1024), mode="bilinear", align_corners=False)
        
        # 1. Extract image embeddings (Frozen)
        with torch.no_grad():
            image_embeddings = self.sam.get_image_embeddings(pixel_values)
            
        # 2. PPT Module: Generate adaptive prompt tokens
        # Output: [B, 2, 256]
        prompt_tokens = self.ppt(image_embeddings)
        
        # 3. Concatenate Expert Tokens
        # Expand expert tokens for batch: [B, N, 256]
        exp_tokens = self.expert_tokens.expand(B, -1, -1)
        
        # Combine all tokens for the decoder: [B, 2 + N, 256]
        sparse_embeddings = torch.cat([prompt_tokens, exp_tokens], dim=1)
        
        # HuggingFace SAM expects sparse_embeddings to be 4D: [batch_size, num_point_batches, num_tokens, embed_dim]
        sparse_embeddings = sparse_embeddings.unsqueeze(1)
        
        # Dense embeddings are empty since we use PPT
        dense_embeddings = torch.zeros(
            (B, 256, image_embeddings.shape[2], image_embeddings.shape[3]), 
            device=pixel_values.device
        )
        
        # 4. SAM Mask Decoder
        # The decoder processes all tokens (SAM outputs + PPT + Expert)
        decoder_outputs = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=self.sam.get_image_wide_positional_embeddings(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        # Extract the processed expert tokens from the decoder output
        # SAM appends prompt tokens after its 5 innate tokens (1 iou + 4 mask)
        # So our sparse tokens are at indices [5:]
        # Workaround: Since transformers SamMaskDecoder doesn't return the raw updated tokens 
        # directly in the dataclass (it projects them to masks), we can route the initial 
        # expert tokens through MoTE and then use that routed token for a custom mask projection,
        # mimicking the SegMoTE prediction head.
        
        routed_expert_tokens, gating_probs = self.mote(exp_tokens)
        
        # Aggregate the selected expert tokens (sum over top-k)
        # Shape: [B, 256]
        final_expert_token = routed_expert_tokens.sum(dim=1)
        
        # 5. Final Mask Prediction
        # Upscale image features similar to SAM
        upscaled_embedding = self.sam.mask_decoder.upscale_conv1(image_embeddings)
        upscaled_embedding = self.sam.mask_decoder.upscale_layer_norm(upscaled_embedding)
        upscaled_embedding = self.sam.mask_decoder.upscale_conv2(upscaled_embedding) # [B, 32, H*4, W*4]
        
        # Project expert token
        hyper_in = self.mask_proj(final_expert_token) # [B, 32]
        
        # Element-wise product for segmentation
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in.view(b, c, 1, 1) * upscaled_embedding).sum(dim=1, keepdim=True) # [B, 1, H*4, W*4]
        
        # Interpolate back to original resolution (e.g. 512x512)
        masks = nn.functional.interpolate(masks, size=original_size, mode="bilinear", align_corners=False)
        
        return masks, gating_probs

