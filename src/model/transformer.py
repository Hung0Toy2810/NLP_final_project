# =============================================================================
# transformer.py — Pre-LN Transformer Encoder (Tự implement từ đầu)
# =============================================================================
# Implement 100% bằng PyTorch thuần.
#
# Bài báo tham khảo:
#   [1] Vaswani et al., "Attention Is All You Need", NeurIPS 2017
#       → Transformer Encoder Block: Self-Attention + FFN + Residual + LayerNorm
#       → FFN: 2 linear layers + activation: FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂
#
#   [2] Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020
#       → Pre-LN: LayerNorm TRƯỚC sub-layer thay vì SAU
#       → Gradient ổn định hơn, giảm phụ thuộc vào warmup
#       → Post-LN (BERT gốc): gradient bùng nổ ở lớp trên cùng khi khởi tạo
#
#   [3] Glorot & Bengio, AISTATS 2010
#       → Xavier Uniform Init cho FFN layers
#
#   [4] GELU activation — được BERT sử dụng rộng rãi
#       → GELU(x) = x · Φ(x), với Φ là CDF của phân phối chuẩn
#       → Mượt hơn ReLU, cho phép gradient chảy qua vùng âm
# =============================================================================

import torch
import torch.nn as nn
from typing import Optional
from model.attention import MultiHeadSelfAttention


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network — Vaswani et al., NeurIPS 2017, Section 3.3.

    Công thức:
        FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂

    Cấu hình:
        d_model = 768, d_ff = 3072 (tỷ lệ 4×), output = 768
        → Expand: 768 → 3072 (tăng khả năng biểu diễn)
        → Contract: 3072 → 768 (nén lại)

    Ý nghĩa (trích paper):
        "While the linear transformations are the same across different positions,
         they use different parameters from layer to layer."
        → Mỗi position được xử lý ĐỘC LẬP (point-wise), nhưng chia sẻ tham số
          giữa các positions TRONG CÙNG 1 layer.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Expand: d_model → d_ff (768 → 3072)
        self.linear1 = nn.Linear(d_model, d_ff)

        # Contract: d_ff → d_model (3072 → 768)
        self.linear2 = nn.Linear(d_ff, d_model)

        # GELU activation — mượt hơn ReLU
        # GELU(x) = x · Φ(x), Φ = CDF phân phối chuẩn
        # Được BERT, GPT-2 sử dụng rộng rãi
        self.activation = nn.GELU()

        # Dropout sau activation — regularization
        self.dropout = nn.Dropout(dropout)

        # Khởi tạo Xavier — Glorot & Bengio, AISTATS 2010
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # x (B, L, 768) → expand (B, L, 3072) → GELU → contract (B, L, 768)
        x = self.linear1(x)       # (B, L, d_ff=3072)
        x = self.activation(x)    # GELU activation
        x = self.dropout(x)
        x = self.linear2(x)       # (B, L, d_model=768)
        return x


class PreLNEncoderBlock(nn.Module):
    """
    Pre-LN Transformer Encoder Block — Xiong et al., ICML 2020.

    So sánh Post-LN (BERT gốc) vs Pre-LN (của chúng ta):

    Post-LN (BERT):                    Pre-LN (SWFT):
    ┌──────────────┐                   ┌──────────────┐
    │     Input     │                   │     Input     │
    │       ↓       │                   │       ↓       │
    │  Self-Attn    │                   │  LayerNorm    │  ← LN TRƯỚC
    │       ↓       │                   │       ↓       │
    │  + Residual   │                   │  Self-Attn    │
    │       ↓       │                   │       ↓       │
    │  LayerNorm    │  ← LN SAU         │  + Residual   │
    │       ↓       │                   │       ↓       │
    │     FFN       │                   │  LayerNorm    │  ← LN TRƯỚC
    │       ↓       │                   │       ↓       │
    │  + Residual   │                   │     FFN       │
    │       ↓       │                   │       ↓       │
    │  LayerNorm    │  ← LN SAU        │  + Residual   │
    └──────────────┘                   └──────────────┘

    Tại sao Pre-LN tốt hơn cho train from scratch?
    (Xiong et al., ICML 2020, Theorem 1):
        "In Post-LN, the expected gradient norm near the output layer
         is large at initialization, requiring careful learning rate warmup."
        → Pre-LN: gradient ổn định ngay từ đầu, không cần warmup cầu kỳ.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()

        # LayerNorm 1 (trước Self-Attention) — Pre-LN, Xiong et al., ICML 2020
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Multi-Head Self-Attention — Vaswani et al., NeurIPS 2017
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Dropout sau attention (trước residual)
        self.dropout1 = nn.Dropout(dropout)

        # LayerNorm 2 (trước FFN) — Pre-LN
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Feed-Forward Network — Vaswani et al., NeurIPS 2017
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        # Dropout sau FFN (trước residual)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pre-LN Encoder Block:
            x → LN → Attention → Dropout → + x (residual)
            → LN → FFN → Dropout → + x (residual)

        Args:
            x:    (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len)

        Returns:
            (batch_size, seq_len, d_model)
        """
        # ===== Sub-layer 1: Self-Attention =====
        residual = x
        x = self.norm1(x)                         # Pre-LN: normalize TRƯỚC
        x = self.self_attention(x, mask)           # Multi-Head Self-Attention
        x = self.dropout1(x)
        x = x + residual                           # Residual connection

        # ===== Sub-layer 2: Feed-Forward Network =====
        residual = x
        x = self.norm2(x)                         # Pre-LN: normalize TRƯỚC
        x = self.ffn(x)                           # FFN: Expand → GELU → Contract
        x = self.dropout2(x)
        x = x + residual                           # Residual connection

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Pre-LN Encoder Blocks.

    SWFT: 6 layers thay vì 12 (BERT) → giảm FLOPs 2× nhưng giữ chiều rộng 768d.

    Thêm một LayerNorm cuối cùng (final_norm) cho Pre-LN architecture.
    Lý do: Trong Pre-LN, output của block cuối cùng CHƯA được normalize.
    (Xiong et al., ICML 2020, Section 2: "We add an additional layer normalization
     after the last encoder block.")
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()

        # Stack of Pre-LN Encoder Blocks
        self.layers = nn.ModuleList([
            PreLNEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm — cần thiết cho Pre-LN architecture
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Đẩy input qua lần lượt từng Encoder Block.

        Args:
            x:    (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len)

        Returns:
            (batch_size, seq_len, d_model) — hidden states đã qua tất cả layers
        """
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.final_norm(x)

        return x
