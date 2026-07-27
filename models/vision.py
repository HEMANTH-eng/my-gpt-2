from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gpt import GPT
from utils.logger import get_logger

logger = get_logger("vision")


class VisionEncoder(nn.Module):
    """Lightweight Vision Encoder projecting image RGB inputs to LLM embedding dimension d_model."""

    def __init__(self, in_channels: int = 3, d_model: int = 128, patch_size: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # Patch projection convolutional layer
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Projects images of shape (B, C, H, W) to visual token sequence (B, N_patches, d_model)."""
        # images: (B, 3, H, W)
        x = self.proj(images)  # (B, d_model, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, d_model)
        x = self.norm(x)
        return x


class MultimodalGPT(nn.Module):
    """Multimodal GPT model combining Vision Feature Projection with Text Transformer Backbone."""

    def __init__(self, gpt_model: GPT, in_channels: int = 3, patch_size: int = 16) -> None:
        super().__init__()
        self.gpt = gpt_model
        self.vision_encoder = VisionEncoder(
            in_channels=in_channels,
            d_model=self.gpt.config.d_model,
            patch_size=patch_size,
        )

    def forward(
        self,
        idx: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass integrating image sequence tokens and text sequence tokens."""
        if images is None:
            return self.gpt(idx, targets=targets)

        # 1. Extract visual tokens (B, N_patches, d_model)
        visual_embeds = self.vision_encoder(images)

        # 2. Compute text embeddings (B, T_text, d_model)
        text_embeds = self.gpt.embedding(idx)

        # 3. Concatenate visual embeddings + text embeddings along sequence dim (dim=1)
        combined_embeds = torch.cat((visual_embeds, text_embeds), dim=1)

        # Truncate sequence length if exceeding model maximum context length
        max_len = self.gpt.config.max_seq_len
        if combined_embeds.size(1) > max_len:
            combined_embeds = combined_embeds[:, -max_len:, :]

        # 4. Pass through Transformer Blocks Stack
        x = combined_embeds
        for block in self.gpt.blocks:
            x = block(x)

        x = self.gpt.ln_f(x)
        logits = self.gpt.lm_head(x)

        loss = None
        if targets is not None:
            # Shift targets calculation if needed
            target_len = targets.size(1)
            logits_slice = logits[:, -target_len:, :]
            loss = F.cross_entropy(logits_slice.reshape(-1, logits_slice.size(-1)), targets.reshape(-1))

        return logits, loss
