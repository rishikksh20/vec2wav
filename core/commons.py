import math

import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from core.attention import ConformerMultiHeadedSelfAttention

LRELU_SLOPE = 0.3


class ConvTransposed(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            padding: int = 0,
    ):
        super().__init__()
        self.conv = BSConv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x


def positional_encoding(
        d_model: int, length: int, device: torch.device
) -> torch.Tensor:
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


class GLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=1)
        x = out * self.lrelu(gate)
        return x


class DepthWiseConv1d(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, groups=in_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def calc_same_padding(kernel_size: int):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            d_model: int,
            expansion_factor: int = 2,
            kernel_size: int = 7,
            dropout: float = 0.1,
    ):
        super().__init__()
        inner_dim = d_model * expansion_factor
        self.ln_1 = nn.LayerNorm(d_model)
        self.conv_1 = PointwiseConv1d(d_model, inner_dim * 2)
        self.conv_act = GLUActivation()
        self.depthwise = DepthWiseConv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=calc_same_padding(kernel_size)[0],
        )
        self.ln_2 = nn.GroupNorm(1, inner_dim)
        self.activation = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_2 = PointwiseConv1d(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(x)
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = self.conv_act(x)
        x = self.depthwise(x)
        x = self.ln_2(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x


class DepthwiseConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        padding = calc_same_padding(kernel_size)
        self.depthwise = nn.Conv1d(
            dim,
            dim * expansion,
            kernel_size=kernel_size,
            padding=padding[0],
            groups=dim,
        )
        self.act = nn.LeakyReLU(LRELU_SLOPE)
        self.out = nn.Conv1d(dim * expansion, dim, 1, 1, 0)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x.permute((0, 2, 1))
        x = self.depthwise(x)
        x = self.act(x)
        x = self.out(x)
        x = x.permute((0, 2, 1))
        return x


class BSConv1d(nn.Module):
    """https://arxiv.org/pdf/2003.13549.pdf"""

    def __init__(
            self, channels_in: int, channels_out: int, kernel_size: int, padding: int
    ):
        super().__init__()
        self.pointwise = nn.Conv1d(channels_in, channels_out, kernel_size=1)
        self.depthwise = nn.Conv1d(
            channels_out,
            channels_out,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.pointwise(x)
        x2 = self.depthwise(x1)
        return x2


class Conv1dGLU(nn.Module):
    """From DeepVoice 3"""

    def __init__(
            self, d_model: int, kernel_size: int, padding: int, embedding_dim: int
    ):
        super().__init__()
        self.conv = BSConv1d(
            d_model, 2 * d_model, kernel_size=kernel_size, padding=padding
        )
        self.embedding_proj = nn.Linear(embedding_dim, d_model)
        self.register_buffer("sqrt", torch.sqrt(torch.FloatTensor([0.5])).squeeze(0))
        self.softsign = torch.nn.Softsign()

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        x = x.permute((0, 2, 1))
        residual = x
        x = self.conv(x)
        splitdim = 1
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        embeddings = self.embedding_proj(embeddings)
        softsign = self.softsign(embeddings)
        a = a + softsign.permute((0, 2, 1))
        x = a * torch.sigmoid(b)
        x = x + residual
        x = x * self.sqrt
        x = x.permute((0, 2, 1))
        return x


class FeedForward(nn.Module):
    def __init__(
            self, d_model: int, kernel_size: int, dropout: float, expansion_factor: int = 4
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.conv_1 = nn.Conv1d(
            d_model,
            d_model * expansion_factor,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_2 = nn.Conv1d(d_model * expansion_factor, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x.permute((0, 2, 1))
        x = self.conv_1(x)
        x = x.permute((0, 2, 1))
        x = self.act(x)
        x = self.dropout(x)
        x = x.permute((0, 2, 1))
        x = self.conv_2(x)
        x = x.permute((0, 2, 1))
        x = self.dropout(x)
        x = 0.5 * x
        return x


class AddCoords(nn.Module):
    def __init__(self, rank: int, with_r: bool = False):
        super().__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = x.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            xx_channel = xx_channel.to(x.device)
            out = torch.cat([x, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = x.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            xx_channel = xx_channel.to(x.device)
            yy_channel = yy_channel.to(x.device)

            out = torch.cat([x, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(
                    torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2)
                )
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = x.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

            xx_channel = xx_channel.to(x.device)
            yy_channel = yy_channel.to(x.device)
            zz_channel = zz_channel.to(x.device)
            out = torch.cat([x, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(
                    torch.pow(xx_channel - 0.5, 2)
                    + torch.pow(yy_channel - 0.5, 2)
                    + torch.pow(zz_channel - 0.5, 2)
                )
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv1d(conv.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            with_r: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv1d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class CoordConv2d(conv.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            with_r: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class ConformerBlock(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            d_k: int,
            d_v: int,
            kernel_size_conv_mod: int,
            embedding_dim: int,
            dropout: float,
            with_ff: bool,
    ):
        super().__init__()
        self.with_ff = with_ff
        self.conditioning = Conv1dGLU(
            d_model=d_model,
            kernel_size=kernel_size_conv_mod,
            padding=kernel_size_conv_mod // 2,
            embedding_dim=embedding_dim,
        )
        if self.with_ff:
            self.ff = FeedForward(d_model=d_model, dropout=dropout, kernel_size=3)
        self.conformer_conv_1 = ConformerConvModule(
            d_model, kernel_size=kernel_size_conv_mod, dropout=dropout
        )
        self.ln = nn.LayerNorm(d_model)
        self.slf_attn = ConformerMultiHeadedSelfAttention(
            d_model=d_model, num_heads=n_head, dropout_p=dropout
        )
        self.conformer_conv_2 = ConformerConvModule(
            d_model, kernel_size=kernel_size_conv_mod, dropout=dropout
        )

    def forward(
            self,
            x: torch.Tensor,
            embeddings: torch.Tensor,
            mask: torch.Tensor,
            slf_attn_mask: torch.Tensor,
            encoding: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings is not None:
            x = self.conditioning(x, embeddings=embeddings)
        if self.with_ff:
            x = self.ff(x) + x
        x = self.conformer_conv_1(x) + x
        res = x
        x = self.ln(x)
        x, _ = self.slf_attn(
            query=x, key=x, value=x, mask=slf_attn_mask, encoding=encoding
        )
        x = x + res
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = self.conformer_conv_2(x) + x
        return x