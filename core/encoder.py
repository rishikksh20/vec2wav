# Semantic Encoder
import torch
from torch import nn
from core.commons import ConformerBlock

class SemanticEncoder(nn.Module):
    # Cross-Attention Conformer
    def __init__(
            self,
            dim: int,
            n_layers: int,
            n_heads: int,
            embedding_dim: int,
            p_dropout: float,
            kernel_size_conv_mod: int,
            with_ff: bool,
    ):
        super().__init__()
        d_k = d_v = dim // n_heads
        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    dim,
                    n_heads,
                    d_k,
                    d_v,
                    kernel_size_conv_mod=kernel_size_conv_mod,
                    dropout=p_dropout,
                    embedding_dim=embedding_dim,
                    with_ff=with_ff,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            embeddings: torch.Tensor,
            encoding: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = mask.view((mask.shape[0], 1, 1, mask.shape[1]))

        output_list = []
        for enc_layer in self.layer_stack:
            x = enc_layer(
                x,
                mask=mask,
                slf_attn_mask=attn_mask,
                embeddings=embeddings,
                encoding=encoding,
            )
            output_list.append(x)
        return x, output_list