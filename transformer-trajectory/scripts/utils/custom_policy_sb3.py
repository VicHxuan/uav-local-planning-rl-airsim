from tracemalloc import start
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
    """
    MLP 特征提取器（用于消融实验）
    
    架构流程：
    1. 输入：[batch, frame_history * single_frame_dim] = [batch, 160]
       - 4 帧历史，每帧 40 维（36 LiDAR + 2 goal + 2 UAV）
    
    2. 分帧并嵌入：
       - 每帧的 lidar、goal、uav 分别通过独立的 Linear 层映射到 d_model 维
       - 每帧生成 3 个独立的 token
       - 总共生成 frame_history * 3 = 12 个 token
    
    3. 全局池化：
       - 对所有 token 进行平均池化
    
    4. 输出投影：
       - 通过线性层映射到 features_dim
    
    相比 MLP_embedding_Transformer_encoder，移除了：
    - 位置编码
    - Transformer Encoder 层
    - 因果掩码
    
    用于验证 Transformer 模块的贡献
    """
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        lidar_feature_dim: int = 64, # 默认值改为 64
        goal_feature_dim: int = 4,
        uav_feature_dim: int = 12,    # 默认值改为 9
        frame_history: int = 4,
        d_model: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        
        # 保存配置参数
        self.lidar_feature_dim = lidar_feature_dim
        self.goal_feature_dim = goal_feature_dim
        self.uav_feature_dim = uav_feature_dim
        self.frame_history = frame_history
        self.single_frame_dim = lidar_feature_dim + goal_feature_dim + uav_feature_dim  # 40
        self.d_model = d_model
        self.tokens_per_frame = 3  # 每帧生成 3 个 token（lidar, goal, uav）
        
        # 验证输入维度
        full_dim = int(np.prod(observation_space.shape))
        expected_dim = self.single_frame_dim * frame_history
        assert full_dim == expected_dim, \
            f"观测空间维度 {full_dim} 不匹配预期 {expected_dim}"
        
        # ===== 1. MLP Embedding 层 =====
        # 为 lidar、goal、uav 各自设计独立的 embedding 层
        # 直接映射到 d_model 维度
        self.lidar_embedding = nn.Linear(lidar_feature_dim, d_model)
        self.goal_embedding = nn.Linear(goal_feature_dim, d_model)
        self.uav_embedding = nn.Linear(uav_feature_dim, d_model)
        
        # ===== 2. 中间处理层（可选）=====
        # 添加 Layer Normalization 和 Dropout 增强鲁棒性
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # ===== 3. 输出投影层 =====
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        前向传播
        
        Args:
            observations: [batch_size, frame_history * single_frame_dim]
                         即 [batch_size, 160]
        Returns:
            features: [batch_size, features_dim]
        """
        batch_size = observations.size(0)
        
        # ===== Step 1: 重塑输入 =====
        # [batch, 160] → [batch, frame_history, single_frame_dim]
        # [batch, 160] → [batch, 4, 40]
        x = observations.reshape(batch_size, self.frame_history, self.single_frame_dim)
        
        # ===== Step 2: 分帧提取并嵌入 =====
        # 为每帧生成 3 个独立的 token（lidar, goal, uav）
        all_tokens = []
        for frame_idx in range(self.frame_history):
            frame = x[:, frame_idx, :]  # [batch, 40]
            
            # 切分特征
            lidar = frame[:, :self.lidar_feature_dim]  # [batch, 36]
            goal = frame[:, self.lidar_feature_dim:self.lidar_feature_dim + self.goal_feature_dim]  # [batch, 2]
            uav = frame[:, self.lidar_feature_dim + self.goal_feature_dim:]  # [batch, 2]
            
            # 分别嵌入到 d_model 维空间
            lidar_token = self.lidar_embedding(lidar).unsqueeze(1)  # [batch, 1, d_model]
            goal_token = self.goal_embedding(goal).unsqueeze(1)     # [batch, 1, d_model]
            uav_token = self.uav_embedding(uav).unsqueeze(1)        # [batch, 1, d_model]
            
            # 按顺序（lidar, goal, uav）添加到 token 列表
            all_tokens.extend([lidar_token, goal_token, uav_token])
        
        # 拼接所有 token: [batch, 12, d_model]
        x_embedded = th.cat(all_tokens, dim=1)
        
        # ===== Step 3: 应用 Layer Norm 和 Dropout =====
        x_embedded = self.embedding_norm(x_embedded)
        x_embedded = self.embedding_dropout(x_embedded)
        
        # ===== Step 4: 全局平均池化 =====
        # 聚合所有 token 的信息
        x_pooled = x_embedded.mean(dim=1)  # [batch, d_model]
        
        # ===== Step 5: 输出投影 =====
        output = self.output_projection(x_pooled)  # [batch, features_dim]
        
        return output

class HierarchicalPositionalEncoding(nn.Module):
    """
    层次化位置编码（改进版）：
    - 帧间位置编码（Frame-level）：标识不同帧的顺序
    - 帧内位置编码（Token-level）：标识帧内不同元素（lidar, goal, uav）的顺序
    
    编码方式：
    - 使用标准的 sinusoidal 位置编码
    - 帧间和帧内编码各自使用完整的 d_model 维度
    - 通过相加（而非拼接）融合两种位置信息
    
    改进点：
    - 每种位置信息都有完整的表达空间（64维）
    - 符合原始Transformer的加法设计理念
    - 表达能力更强，理论基础更扎实
    """
    def __init__(self, d_model: int, max_frames: int = 10, max_tokens_per_frame: int = 10):
        super().__init__()
        self.d_model = d_model
        
        # 帧间位置编码（标识第几帧）- 完整的 d_model 维度
        self.frame_pe = self._generate_positional_encoding(max_frames, d_model)
        
        # 帧内位置编码（标识帧内第几个 token）- 完整的 d_model 维度
        self.token_pe = self._generate_positional_encoding(max_tokens_per_frame, d_model)
        
    def _generate_positional_encoding(self, max_len: int, d_model: int) -> th.Tensor:
        """
        生成标准的 sinusoidal 位置编码
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            max_len: 最大序列长度
            d_model: 编码维度
        Returns:
            pe: [max_len, d_model] 的位置编码矩阵
        """
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float32).unsqueeze(1)  # [max_len, 1]
        
        # 计算分母项：10000^(2i/d_model)
        div_term = th.exp(th.arange(0, d_model, 2, dtype=th.float32) * 
                         (-math.log(10000.0) / d_model))
        
        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        
        return pe
    
    def forward(self, x: th.Tensor, tokens_per_frame: int) -> th.Tensor:
        """
        为输入序列添加层次化位置编码
        
        Args:
            x: [batch, seq_len, d_model] 输入特征
            tokens_per_frame: 每帧包含的 token 数量（本例中为 3）
        Returns:
            x + positional_encoding: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        num_frames = seq_len // tokens_per_frame
        
        # 确保编码在正确的设备上
        device = x.device
        frame_pe = self.frame_pe.to(device)
        token_pe = self.token_pe.to(device)
        
        # 构建完整的位置编码
        pe_full = []
        for frame_idx in range(num_frames):
            # 帧间编码：对该帧的所有 token 使用相同的帧级编码
            # [d_model] → [1, d_model] → [tokens_per_frame, d_model]
            frame_encoding = frame_pe[frame_idx].unsqueeze(0).repeat(tokens_per_frame, 1)
            
            # 帧内编码：标识该帧内各 token 的相对位置
            # [tokens_per_frame, d_model]
            token_encoding = token_pe[:tokens_per_frame]

            combined = frame_encoding + token_encoding
            
            pe_full.append(combined)
        
        # 拼接所有帧的位置编码
        pe_full = th.cat(pe_full, dim=0)  # [seq_len, d_model]
        pe_full = pe_full.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, seq_len, d_model]
        
        return x + pe_full



class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制（支持因果掩码）
    
    核心计算：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    特性：
    - 支持多头并行计算
    - 支持因果掩码（用于序列建模）
    - 使用 Scaled Dot-Product Attention
    """
    def __init__(self, d_model: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: th.Tensor, mask: th.Tensor = None) -> th.Tensor:
        """
        多头自注意力前向传播
        
        Args:
            x: [batch, seq_len, d_model] 输入特征
            mask: [batch, seq_len, seq_len] or [1, seq_len, seq_len] 注意力掩码
                  mask[i, j] = 0 表示位置 i 不能关注位置 j
        Returns:
            output: [batch, seq_len, d_model] 输出特征
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性投影并分头
        # [batch, seq_len, d_model] → [batch, seq_len, num_heads, d_k] → [batch, num_heads, seq_len, d_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # scores: [batch, num_heads, seq_len, seq_len]
        scores = th.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（将被掩盖的位置设为极小值，使 softmax 后接近 0）
        if mask is not None:
            # 如果 mask 是 [1, seq_len, seq_len]，会自动广播到 [batch, num_heads, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = th.matmul(attn_weights, V)  # [batch, num_heads, seq_len, d_k]
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.W_o(context)
        
        return output


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    结构：
    FFN(x) = ReLU(xW1 + b1)W2 + b2
    
    每个位置独立应用相同的全连接网络
    """
    def __init__(self, d_model: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    标准 Transformer Encoder Layer
    
    结构：
    1. Multi-Head Self-Attention + Dropout + Residual + LayerNorm
    2. Feed-Forward Network + Dropout + Residual + LayerNorm
    
    这是标准的 Post-Norm 结构（残差后归一化）
    """
    def __init__(self, d_model: int, num_heads: int = 2, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Multi-Head Self-Attention 子层
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-Forward Network 子层
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: th.Tensor, mask: th.Tensor = None) -> th.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len] 注意力掩码
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Self-Attention + Residual + LayerNorm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # FFN + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x


class MLP_embedding_Transformer_encoder(BaseFeaturesExtractor):
    """
    MLP Embedding + Transformer Encoder 特征提取器（带因果掩码）
    
    架构流程：
    1. 输入：[batch, frame_history * single_frame_dim] = [batch, 160]
       - 4 帧历史，每帧 40 维（36 LiDAR + 2 goal + 2 UAV）
    
    2. 分帧并嵌入：
       - 每帧的 lidar、goal、uav 分别通过独立的 Linear 层映射到 d_model 维
       - 每帧生成 3 个独立的 token
       - 总共生成 frame_history * 3 = 12 个 token
    
    3. 层次化位置编码：
       - 帧间编码：标识第几帧（0-3）
       - 帧内编码：标识帧内第几个 token（0-2，对应 lidar、goal、uav）
    
    4. 因果掩码：
       - 确保早期帧不能关注后续帧
       - 同一帧内的 token 可以相互关注
       - 掩码矩阵为上三角矩阵（按帧级别）
    
    5. Transformer Encoder：
       - num_layers 层 Encoder
       - 每层包含 Multi-Head Self-Attention 和 FFN
       - 使用 LayerNorm 和残差连接
    
    6. 全局池化 + 输出投影：
       - 对所有 token 进行平均池化
       - 通过线性层映射到 features_dim
    """
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        lidar_feature_dim: int = 36,
        goal_feature_dim: int = 2,
        uav_feature_dim: int = 2,
        frame_history: int = 4,
        d_model: int = 64,           # Transformer 隐藏维度
        num_heads: int = 2,          # 注意力头数
        num_layers: int = 2,         # Encoder 层数
        d_ff: int = 256,             # FFN 中间层维度
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        
        # 保存配置参数
        self.lidar_feature_dim = lidar_feature_dim
        self.goal_feature_dim = goal_feature_dim
        self.uav_feature_dim = uav_feature_dim
        self.frame_history = frame_history
        self.single_frame_dim = lidar_feature_dim + goal_feature_dim + uav_feature_dim  # 40
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.tokens_per_frame = 3  # 每帧生成 3 个 token（lidar, goal, uav）
        
        # 验证输入维度
        full_dim = int(np.prod(observation_space.shape))
        expected_dim = self.single_frame_dim * frame_history
        assert full_dim == expected_dim, \
            f"观测空间维度 {full_dim} 不匹配预期 {expected_dim}"
        
        # ===== 1. MLP Embedding 层 =====
        # 为 lidar、goal、uav 各自设计独立的 embedding 层
        # 直接映射到 d_model 维度
        self.lidar_embedding = nn.Linear(lidar_feature_dim, d_model)
        self.goal_embedding = nn.Linear(goal_feature_dim, d_model)
        self.uav_embedding = nn.Linear(uav_feature_dim, d_model)
        
        # ===== 2. 层次化位置编码 =====
        self.positional_encoding = HierarchicalPositionalEncoding(
            d_model=d_model,
            max_frames=frame_history + 5,  # 留些余量
            max_tokens_per_frame=self.tokens_per_frame + 2
        )
        
        # ===== 3. Transformer Encoder 层 =====
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # ===== 4. 输出投影层 =====
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        
        # ===== 5. 预生成因果掩码（用于推理加速）=====
        # 掩码形状：[1, total_tokens, total_tokens]
        # 因果约束：早期帧不能关注后续帧
        self.register_buffer(
            "causal_mask",
            self._generate_frame_causal_mask(frame_history, self.tokens_per_frame)
        )
        
    def _generate_frame_causal_mask(self, num_frames: int, tokens_per_frame: int) -> th.Tensor:
        """
        生成帧级别的因果掩码
        
        规则：
        - 同一帧内的 token 可以相互关注
        - 早期帧的 token 可以关注自己和更早帧的所有 token
        - 早期帧的 token 不能关注后续帧的 token
        
        Args:
            num_frames: 帧数（4）
            tokens_per_frame: 每帧的 token 数量（3）
        Returns:
            mask: [1, total_tokens, total_tokens] 因果掩码矩阵
        """
        total_tokens = num_frames * tokens_per_frame
        mask = th.zeros(total_tokens, total_tokens, dtype=th.bool)
        
        # 按帧构建掩码
        for query_frame in range(num_frames):
            for key_frame in range(num_frames):
                # 只有当 key_frame <= query_frame 时才能关注
                if key_frame <= query_frame:
                    # 计算帧内的 token 索引范围
                    query_start = query_frame * tokens_per_frame
                    query_end = (query_frame + 1) * tokens_per_frame
                    key_start = key_frame * tokens_per_frame
                    key_end = (key_frame + 1) * tokens_per_frame
                    
                    # 该帧的所有 token 可以关注更早帧的所有 token
                    mask[query_start:query_end, key_start:key_end] = True
        
        # 添加 batch 维度并转为 float（掩码约定：1 表示可关注，0 表示屏蔽）
        return mask.unsqueeze(0).float()  # [1, total_tokens, total_tokens]
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        前向传播
        
        Args:
            observations: [batch_size, frame_history * single_frame_dim]
                         即 [batch_size, 160]
        Returns:
            features: [batch_size, features_dim]
        """
        batch_size = observations.size(0)
        device = observations.device
        
        # ===== Step 1: 重塑输入 =====
        # [batch, 160] → [batch, frame_history, single_frame_dim]
        # [batch, 160] → [batch, 4, 40]
        x = observations.reshape(batch_size, self.frame_history, self.single_frame_dim)
        
        # ===== Step 2: 分帧提取并嵌入 =====
        # 为每帧生成 3 个独立的 token（lidar, goal, uav）
        all_tokens = []
        for frame_idx in range(self.frame_history):
            frame = x[:, frame_idx, :]  # [batch, 40]
            
            # 切分特征
            lidar = frame[:, :self.lidar_feature_dim]  # [batch, 64]
            goal = frame[:, self.lidar_feature_dim:self.lidar_feature_dim + self.goal_feature_dim]  # [batch, 4]
            uav = frame[:, self.lidar_feature_dim + self.goal_feature_dim:]  # [batch, 12]           
            # 分别嵌入到 d_model 维空间
            lidar_token = self.lidar_embedding(lidar).unsqueeze(1)  # [batch, 1, d_model]
            goal_token = self.goal_embedding(goal).unsqueeze(1)     # [batch, 1, d_model]
            uav_token = self.uav_embedding(uav).unsqueeze(1)        # [batch, 1, d_model]
            
            # 按顺序（lidar, goal, uav）添加到 token 列表
            all_tokens.extend([lidar_token, goal_token, uav_token])
        
        # 拼接所有 token: [batch, 12, d_model]
        x_embedded = th.cat(all_tokens, dim=1)
        
        # ===== Step 3: 添加位置编码 =====
        x_embedded = self.positional_encoding(x_embedded, tokens_per_frame=self.tokens_per_frame)
        
        # ===== Step 4: 准备因果掩码 =====
        # 掩码已在 __init__ 中预生成，这里直接使用
        # 需要扩展到当前 batch_size 和 num_heads 维度
        # 原始掩码：[1, seq_len, seq_len]
        # MultiHeadAttention 需要：[batch, num_heads, seq_len, seq_len] 或 [1, seq_len, seq_len]
        # 我们使用 [1, seq_len, seq_len]，会自动广播
        causal_mask = self.causal_mask.to(device)  # [1, 12, 12]
        
        # ===== Step 5: 通过 Transformer Encoder =====
        for encoder_layer in self.encoder_layers:
            x_embedded = encoder_layer(x_embedded, mask=causal_mask)  # [batch, 12, d_model]
        
        # ===== Step 6: 全局池化 + 输出投影 =====
        # 使用平均池化聚合所有 token 的信息
        x_pooled = x_embedded.mean(dim=1)  # [batch, d_model]
        
        # 投影到最终特征维度
        output = self.output_projection(x_pooled)  # [batch, features_dim]
        
        return output
