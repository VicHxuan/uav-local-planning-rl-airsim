"""SB3 policy components: feature extractors used by the training scripts."""

import gym
import torch.nn as nn
import torch as th
from torch.nn.modules.linear import Linear
import math
import torchvision.models as pre_models
import numpy as np
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MLP_encoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        lidar_feature_dim: int = 64,
        goal_feature_dim: int = 4,
        uav_feature_dim: int = 12,
        frame_history: int = 4,
        d_model: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        self.lidar_feature_dim = lidar_feature_dim
        self.goal_feature_dim = goal_feature_dim
        self.uav_feature_dim = uav_feature_dim
        self.frame_history = frame_history
        self.single_frame_dim = lidar_feature_dim + goal_feature_dim + uav_feature_dim
        self.d_model = d_model
        self.tokens_per_frame = 3
        full_dim = int(np.prod(observation_space.shape))
        expected_dim = self.single_frame_dim * frame_history
        assert full_dim == expected_dim, \
            f"观测空间维度 {full_dim} 不匹配预期 {expected_dim}"
        self.lidar_embedding = nn.Linear(lidar_feature_dim, d_model)
        self.goal_embedding = nn.Linear(goal_feature_dim, d_model)
        self.uav_embedding = nn.Linear(uav_feature_dim, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size(0)
        x = observations.reshape(batch_size, self.frame_history, self.single_frame_dim)
        all_tokens = []
        for frame_idx in range(self.frame_history):
            frame = x[:, frame_idx, :]
            lidar = frame[:, :self.lidar_feature_dim]
            goal = frame[:, self.lidar_feature_dim:self.lidar_feature_dim + self.goal_feature_dim]
            uav = frame[:, self.lidar_feature_dim + self.goal_feature_dim:]
            lidar_token = self.lidar_embedding(lidar).unsqueeze(1)
            goal_token = self.goal_embedding(goal).unsqueeze(1)
            uav_token = self.uav_embedding(uav).unsqueeze(1)
            all_tokens.extend([lidar_token, goal_token, uav_token])
        x_embedded = th.cat(all_tokens, dim=1)
        x_embedded = self.embedding_norm(x_embedded)
        x_embedded = self.embedding_dropout(x_embedded)
        x_pooled = x_embedded.mean(dim=1)
        output = self.output_projection(x_pooled)
        return output

class HierarchicalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_frames: int = 10, max_tokens_per_frame: int = 10):
        super().__init__()
        self.d_model = d_model
        self.frame_pe = self._generate_positional_encoding(max_frames, d_model)
        self.token_pe = self._generate_positional_encoding(max_tokens_per_frame, d_model)
        
    def _generate_positional_encoding(self, max_len: int, d_model: int) -> th.Tensor:
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float32).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2, dtype=th.float32) * 
                         (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        return pe
    
    def forward(self, x: th.Tensor, tokens_per_frame: int) -> th.Tensor:
        batch_size, seq_len, d_model = x.shape
        num_frames = seq_len // tokens_per_frame
        device = x.device
        frame_pe = self.frame_pe.to(device)
        token_pe = self.token_pe.to(device)
        pe_full = []
        for frame_idx in range(num_frames):
            frame_encoding = frame_pe[frame_idx].unsqueeze(0).repeat(tokens_per_frame, 1)
            token_encoding = token_pe[:tokens_per_frame]
            combined = frame_encoding + token_encoding
            pe_full.append(combined)
        pe_full = th.cat(pe_full, dim=0)
        pe_full = pe_full.unsqueeze(0).expand(batch_size, -1, -1)
        return x + pe_full

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: th.Tensor, mask: th.Tensor = None) -> th.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = th.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = th.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 2, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: th.Tensor, mask: th.Tensor = None) -> th.Tensor:
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class MLP_embedding_Transformer_encoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        lidar_feature_dim: int = 36,
        goal_feature_dim: int = 2,
        uav_feature_dim: int = 2,
        frame_history: int = 4,
        d_model: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        self.lidar_feature_dim = lidar_feature_dim
        self.goal_feature_dim = goal_feature_dim
        self.uav_feature_dim = uav_feature_dim
        self.frame_history = frame_history
        self.single_frame_dim = lidar_feature_dim + goal_feature_dim + uav_feature_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.tokens_per_frame = 3
        full_dim = int(np.prod(observation_space.shape))
        expected_dim = self.single_frame_dim * frame_history
        assert full_dim == expected_dim, \
            f"观测空间维度 {full_dim} 不匹配预期 {expected_dim}"
        self.lidar_embedding = nn.Linear(lidar_feature_dim, d_model)
        self.goal_embedding = nn.Linear(goal_feature_dim, d_model)
        self.uav_embedding = nn.Linear(uav_feature_dim, d_model)
        self.positional_encoding = HierarchicalPositionalEncoding(
            d_model=d_model,
            max_frames=frame_history + 5,
            max_tokens_per_frame=self.tokens_per_frame + 2
        )
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.register_buffer(
            "causal_mask",
            self._generate_frame_causal_mask(frame_history, self.tokens_per_frame)
        )
        
    def _generate_frame_causal_mask(self, num_frames: int, tokens_per_frame: int) -> th.Tensor:
        total_tokens = num_frames * tokens_per_frame
        mask = th.zeros(total_tokens, total_tokens, dtype=th.bool)
        for query_frame in range(num_frames):
            for key_frame in range(num_frames):
                if key_frame <= query_frame:
                    query_start = query_frame * tokens_per_frame
                    query_end = (query_frame + 1) * tokens_per_frame
                    key_start = key_frame * tokens_per_frame
                    key_end = (key_frame + 1) * tokens_per_frame
                    mask[query_start:query_end, key_start:key_end] = True
        return mask.unsqueeze(0).float()
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size(0)
        device = observations.device
        x = observations.reshape(batch_size, self.frame_history, self.single_frame_dim)
        all_tokens = []
        for frame_idx in range(self.frame_history):
            frame = x[:, frame_idx, :]
            lidar = frame[:, :self.lidar_feature_dim]
            goal = frame[:, self.lidar_feature_dim:self.lidar_feature_dim + self.goal_feature_dim]
            uav = frame[:, self.lidar_feature_dim + self.goal_feature_dim:]           
            lidar_token = self.lidar_embedding(lidar).unsqueeze(1)
            goal_token = self.goal_embedding(goal).unsqueeze(1)
            uav_token = self.uav_embedding(uav).unsqueeze(1)
            all_tokens.extend([lidar_token, goal_token, uav_token])
        x_embedded = th.cat(all_tokens, dim=1)
        x_embedded = self.positional_encoding(x_embedded, tokens_per_frame=self.tokens_per_frame)
        causal_mask = self.causal_mask.to(device)
        for encoder_layer in self.encoder_layers:
            x_embedded = encoder_layer(x_embedded, mask=causal_mask)
        x_pooled = x_embedded.mean(dim=1)
        output = self.output_projection(x_pooled)
        return output
