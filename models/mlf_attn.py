import math

import torch
import torch.nn as nn


class MultiLevelForensicAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, kernel_size=3, dilation=(2, 3)):
        super().__init__()
        self.num_dilation = len(dilation)
        self.num_heads = num_heads

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.unfold_layers = nn.ModuleList(
            nn.Unfold(kernel_size, dilation_rate, dilation_rate * (kernel_size - 1) // 2, 1)
            for dilation_rate in dilation
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        batch_size, sequence_length, channels = q.shape

        level_q = self._split_projection(self.q, q, batch_size, sequence_length, channels)
        level_k = self._split_projection(self.k, k, batch_size, sequence_length, channels)
        level_v = self._split_projection(self.v, v, batch_size, sequence_length, channels)

        level_outputs = []
        for unfold, q_slice, k_slice, v_slice in zip(self.unfold_layers, level_q, level_k, level_v):
            _, level_channels, _ = q_slice.shape
            head_count = level_channels // self.head_dim

            query = q_slice.reshape(batch_size, head_count, self.head_dim, 1, sequence_length)
            query = query.permute(0, 1, 4, 3, 2)
            key = self._local_windows(unfold, k_slice, batch_size, head_count, sequence_length).transpose(-1, -2)
            value = self._local_windows(unfold, v_slice, batch_size, head_count, sequence_length)

            attention = query.matmul(key).mul(self.scale).softmax(dim=-1)
            attention = self.attn_drop(attention)
            level_outputs.append(
                attention.matmul(value).transpose(1, 2).reshape(batch_size, sequence_length, level_channels)
            )

        x = torch.stack(level_outputs, dim=-1).reshape(batch_size, sequence_length, channels)
        return self.proj_drop(self.proj(x))

    def _split_projection(self, projection, tokens, batch_size, sequence_length, channels):
        channels_per_level = channels // self.num_dilation
        return projection(tokens).reshape(
            self.num_dilation,
            batch_size,
            sequence_length,
            channels_per_level,
        ).transpose(-1, -2)

    def _local_windows(self, unfold, tokens, batch_size, head_count, sequence_length):
        grid_size = math.isqrt(sequence_length)
        feature_map = tokens.reshape(batch_size, head_count * self.head_dim, grid_size, grid_size)
        return unfold(feature_map).reshape(
            batch_size,
            head_count,
            self.head_dim,
            self.kernel_size * self.kernel_size,
            sequence_length,
        ).permute(0, 1, 4, 3, 2)
