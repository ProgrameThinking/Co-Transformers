import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp as TimmMlp

from models.mlf_attn import MultiLevelForensicAttention

class CrossTraceExtractionTransformer(nn.Module):
    def __init__(self, patch_size=32, input_size=512, num_prototypes=256, embedding_dim=768, num_layers=8):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.random_position_embedding = nn.Parameter(torch.randn(1, (input_size // patch_size) ** 2, embedding_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CrossTraceTransformerBlock(num_prototypes, embedding_dim)
            for _ in range(num_layers)
        ])

    def forward(self, image):
        x_noise = self.patch_embedding(image)
        batch_size, channels, height, width = x_noise.shape
        patch_embeddings = x_noise.flatten(2).transpose(1, 2)
        patch_embeddings = patch_embeddings + self.random_position_embedding

        for block in self.blocks:
            patch_embeddings = block(patch_embeddings)

        refined_patches = patch_embeddings.transpose(1, 2).reshape(batch_size, channels, height, width)
        return F.interpolate(refined_patches, scale_factor=2, mode='bilinear', align_corners=False)


class CrossTraceTransformerBlock(nn.Module):
    def __init__(self, num_prototypes, embedding_dim):
        super().__init__()
        self.object_encoder = CrossTraceEncoder(embedding_dim, num_prototypes=num_prototypes)
        self.patch_decoder = CrossTraceDecoder(embedding_dim)
        self.boundary_modulator = BCIM(embedding_dim)

    def forward(self, patch_embeddings):
        object_features = self.object_encoder(patch_embeddings)
        decoded_patches = self.patch_decoder(patch_embeddings, object_features)
        return self.boundary_modulator(decoded_patches)


class CrossTraceEncoder(nn.Module):
    def __init__(self, dim=768, num_heads=16, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.0, attn_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=TimmMlp, num_prototypes=768):
        super().__init__()
        self.object_prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        self.norm1 = norm_layer(dim)
        self.mlf_attention = MultiLevelForensicAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)

    def forward(self, patch_embeddings):
        batch_size = patch_embeddings.shape[0]
        query = self.norm1(self.object_prototypes).unsqueeze(0).repeat(batch_size, 1, 1)
        key_value = self.norm1(patch_embeddings)
        object_features = query + self.mlf_attention(query, key_value, key_value)
        object_features = object_features + self.mlp(self.norm2(object_features))
        return object_features


class CrossTraceDecoder(nn.Module):
    def __init__(self, dim=768, num_heads=16, mlp_ratio=4.0, qkv_bias=True, proj_drop=0.0, attn_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=TimmMlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mlf_attention = MultiLevelForensicAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)

    def forward(self, patch_embeddings, object_features):
        patch_embeddings = self.norm1(patch_embeddings)
        object_features = self.norm1(object_features)
        decoded = patch_embeddings + self.mlf_attention(patch_embeddings, object_features, object_features)
        return decoded + self.mlp(self.norm2(decoded))


class BCIM(nn.Module):
    def __init__(self, embedding_dim, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.embedding_dim = embedding_dim

    def forward(self, patch_embeddings):
        batch_size, sequence_length, channels = patch_embeddings.shape
        height = width = int(sequence_length ** 0.5)
        feature_map = patch_embeddings.transpose(1, 2).reshape(batch_size, channels, height, width)

        unfolded = F.unfold(feature_map, kernel_size=self.window_size, padding=self.window_size // 2)
        unfolded = unfolded.reshape(batch_size, channels, self.window_size ** 2, height, width)
        cosine_similarity = F.cosine_similarity(feature_map.unsqueeze(2), unfolded, dim=1)
        similarity_matrix = cosine_similarity.mean(dim=1, keepdim=True) / (self.window_size ** 2)
        return (feature_map * similarity_matrix).reshape(batch_size, channels, height * width).transpose(1, 2)
