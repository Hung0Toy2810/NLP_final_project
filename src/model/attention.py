# =============================================================================
# attention.py — Multi-Head Self-Attention (Tự implement từ đầu)
# =============================================================================
# Implement 100% bằng PyTorch thuần (torch.nn, torch.matmul).
#
# Bài báo tham khảo:
#   [1] Vaswani et al., "Attention Is All You Need", NeurIPS 2017
#       → Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
#       → Multi-Head: cho phép mô hình "chú ý" vào nhiều khía cạnh khác nhau song song
#
#   [2] Glorot & Bengio, AISTATS 2010
#       → Xavier Uniform Init cho các ma trận W_Q, W_K, W_V, W_O
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention — Vaswani et al., NeurIPS 2017, Section 3.2.1.

    Công thức:
        Attention(Q, K, V) = softmax( Q · K^T / √d_k ) · V

    Giải thích từng bước:
        1. Q · K^T: Tính "mức độ liên quan" giữa mỗi cặp token (attention scores)
        2. / √d_k : Scaling — ngăn dot product quá lớn khi d_k lớn,
                     khiến softmax bão hòa (gradient ≈ 0)
        3. softmax : Chuẩn hóa scores thành xác suất (tổng = 1)
        4. × V    : Lấy trung bình có trọng số (weighted sum) của Value vectors
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query:  (batch, num_heads, seq_len, d_k)
            key:    (batch, num_heads, seq_len, d_k)
            value:  (batch, num_heads, seq_len, d_k)
            mask:   (batch, 1, 1, seq_len) — 1 cho token thật, 0 cho padding

        Returns:
            output:  (batch, num_heads, seq_len, d_k)
            weights: (batch, num_heads, seq_len, seq_len) — attention weights (cho visualization)
        """
        d_k = query.size(-1)

        # Bước 1: Q · K^T → attention scores
        # (B, H, L, d_k) × (B, H, d_k, L) → (B, H, L, L)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Bước 2: Scale bằng √d_k — ngăn softmax bão hòa
        # (Vaswani et al., NeurIPS 2017, Section 3.2.1:
        #  "We suspect that for large values of d_k, the dot products grow large in magnitude,
        #   pushing the softmax function into regions where it has extremely small gradients.")
        scores = scores / math.sqrt(d_k)

        # Bước 3: Áp dụng mask — đặt score = -inf cho padding tokens
        # Sau softmax, padding tokens sẽ có weight ≈ 0 (e^(-inf) ≈ 0)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Bước 4: Softmax — chuẩn hóa thành xác suất
        attention_weights = F.softmax(scores, dim=-1)

        # Dropout trên attention weights — regularization
        # (SimCSE, EMNLP 2021: dropout đóng vai trò data augmentation ngầm)
        attention_weights = self.dropout(attention_weights)

        # Bước 5: Weighted sum of Values
        # (B, H, L, L) × (B, H, L, d_k) → (B, H, L, d_k)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention — Vaswani et al., NeurIPS 2017, Section 3.2.2.

    Công thức:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
        head_i = Attention(X·W_Q_i, X·W_K_i, X·W_V_i)

    Ý nghĩa (trích paper gốc):
        "Multi-head attention allows the model to jointly attend to information
         from different representation subspaces at different positions."

    Ví dụ với 12 heads:
        - Head 1 có thể học chú ý vào cú pháp (subject-verb agreement)
        - Head 2 có thể học chú ý vào ngữ nghĩa (synonym detection)
        - Head 3 có thể học chú ý vào vị trí tương đối (adjacent tokens)
        - ... và 9 heads khác học các pattern khác

    Cấu hình SWFT:
        d_model = 768, num_heads = 12, d_k = d_v = 768/12 = 64
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) phải chia hết cho num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Chiều của mỗi head = 768/12 = 64

        # Ma trận projection cho Q, K, V — mỗi ma trận shape (d_model, d_model)
        # Thực chất: W_Q = [W_Q_1; W_Q_2; ...; W_Q_h] ghép lại
        # Xavier Uniform Init — Glorot & Bengio, AISTATS 2010
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)

        # Ma trận output projection W_O — (d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model, bias=True)

        # Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Khởi tạo Xavier Uniform — Glorot & Bengio, AISTATS 2010
        self._init_weights()

    def _init_weights(self):
        """
        Khởi tạo trọng số theo Xavier Uniform.
        Glorot & Bengio, "Understanding the difficulty of training deep feedforward
        neural networks", AISTATS 2010:
            W ~ U[-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out))]
        """
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Self-Attention: Q, K, V đều từ cùng input x (vì đây là SELF-attention).

        Args:
            x:    (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) — 1 cho token thật, 0 cho padding

        Returns:
            (batch_size, seq_len, d_model) — output đã qua multi-head attention
        """
        batch_size = x.size(0)

        # ===== Bước 1: Linear projections =====
        # x (B, L, d_model) → Q, K, V (B, L, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # ===== Bước 2: Chia thành nhiều heads =====
        # (B, L, d_model) → (B, L, num_heads, d_k) → (B, num_heads, L, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # ===== Bước 3: Chuẩn bị mask cho attention =====
        if mask is not None:
            # (B, L) → (B, 1, 1, L) — broadcast cho tất cả heads và query positions
            mask = mask.unsqueeze(1).unsqueeze(2)

        # ===== Bước 4: Scaled Dot-Product Attention cho mỗi head =====
        # attention_output: (B, num_heads, L, d_k)
        attention_output, _ = self.attention(Q, K, V, mask)

        # ===== Bước 5: Concat heads =====
        # (B, num_heads, L, d_k) → (B, L, num_heads, d_k) → (B, L, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # ===== Bước 6: Output projection W_O =====
        output = self.W_O(attention_output)  # (B, L, d_model)

        return output
