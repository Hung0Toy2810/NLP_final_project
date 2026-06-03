# =============================================================================
# embedding.py — Factorized Embedding + Sinusoidal Positional Encoding
# =============================================================================
# Implement 100% bằng PyTorch thuần (torch.nn).
#
# Bài báo tham khảo:
#   [1] Lan et al., "ALBERT: A Lite BERT for Self-supervised Learning
#       of Language Representations", ICLR 2020
#       → Factorized Embedding: V×H → V×E + E×H (E << H)
#       → Giảm tham số embedding từ ~23M xuống ~4M
#
#   [2] Vaswani et al., "Attention Is All You Need", NeurIPS 2017
#       → Sinusoidal Positional Encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d))
#       → Không thêm tham số học được, generalize tốt cho mọi sequence length
#
#   [3] Glorot & Bengio, "Understanding the difficulty of training deep
#       feedforward neural networks", AISTATS 2010
#       → Xavier Uniform Initialization cho ma trận embedding
# =============================================================================

import math
import torch
import torch.nn as nn
from typing import cast


class FactorizedEmbedding(nn.Module):
    """
    Factorized Embedding theo ALBERT (Lan et al., ICLR 2020).

    Thay vì ánh xạ trực tiếp Token ID → vector H chiều (30,522 × 768 = ~23M params),
    ta chia làm 2 bước:
        Bước 1: Token ID → vector E chiều nhỏ (30,522 × 128 = ~3.9M params)
        Bước 2: Linear projection E → H       (128 × 768  = ~0.1M params)

    Tổng: ~4M params thay vì ~23M → tiết kiệm ~19M tham số vô nghĩa.

    Lý do khoa học (trích ALBERT paper, Section 3.1):
        "WordPiece embeddings are meant to learn context-INDEPENDENT representations,
         whereas hidden-layer embeddings are meant to learn context-DEPENDENT representations."
        → Không cần embedding dimension lớn bằng hidden dimension.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 layer_norm_eps: float = 1e-12, dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim  # E = 128 (nhỏ)
        self.hidden_size = hidden_size      # H = 768 (lớn)

        # Bước 1: Token ID → E dimensions
        # Khởi tạo Xavier Uniform — Glorot & Bengio, AISTATS 2010
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.token_embedding.weight.data[1:])  # Bỏ qua padding token [0]

        # Bước 2: Linear projection E → H (phóng to lên không gian ngữ nghĩa)
        # Đây chính là "Factorization" — tách V×H thành V×E + E×H
        self.projection = nn.Linear(embedding_dim, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)  # Glorot & Bengio, AISTATS 2010

        # LayerNorm + Dropout sau khi cộng positional encoding
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) — Token IDs từ WordPiece tokenizer

        Returns:
            (batch_size, seq_len, hidden_size) — Embedded vectors sẵn sàng cho Transformer
        """
        # Bước 1: Token ID → E-dimensional vectors (context-independent)
        token_embeds = self.token_embedding(input_ids)  # (B, L, E=128)

        # Bước 2: Project lên H-dimensional space (context-dependent space)
        hidden_states = self.projection(token_embeds)    # (B, L, H=768)

        return hidden_states


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding theo Vaswani et al., NeurIPS 2017.

    Công thức (Equation 3.1 trong paper gốc):
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Ưu điểm so với Learned Positional Embedding:
        1. Không thêm tham số học được → tiết kiệm bộ nhớ
        2. Có thể generalize cho sequence dài hơn lúc training
        3. Khoảng cách tương đối giữa 2 vị trí bất kỳ luôn biểu diễn được
           bằng linear function của positional encodings (Section 3.5 trong paper)
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        # Tạo ma trận PE cố định (không cần gradient)
        pe = torch.zeros(max_len, d_model)     # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # Tính div_term = 1 / 10000^(2i / d_model)
        # Dùng exp(log) để tránh overflow khi d_model lớn
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        # Áp dụng sin cho vị trí chẵn, cos cho vị trí lẻ
        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i) = sin(...)
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1) = cos(...)

        # Thêm batch dimension và đăng ký buffer (không phải parameter)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model) — Output từ FactorizedEmbedding

        Returns:
            (batch_size, seq_len, d_model) — x + positional encoding
        """
        # Cộng PE vào input (broadcast theo batch dimension)
        pe = cast(torch.Tensor, self.get_buffer('pe'))
        return x + pe[:, :x.size(1), :]


class TransformerEmbedding(nn.Module):
    """
    Kết hợp Factorized Embedding + Sinusoidal PE + LayerNorm + Dropout.

    Đây là lớp đầu vào hoàn chỉnh cho Transformer Encoder:
        Token IDs → Factorized Embed (ALBERT) → + Sinusoidal PE → LayerNorm → Dropout
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 max_seq_length: int, layer_norm_eps: float = 1e-12, dropout: float = 0.1):
        super().__init__()

        # Factorized Embedding — Lan et al., ALBERT, ICLR 2020
        self.factorized_embedding = FactorizedEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout
        )

        # Sinusoidal Positional Encoding — Vaswani et al., NeurIPS 2017
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=hidden_size,
            max_len=max_seq_length
        )

        # LayerNorm + Dropout
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            (batch_size, seq_len, hidden_size)
        """
        # 1. Factorized Embedding: Token ID → E → H
        embeddings = self.factorized_embedding(input_ids)  # (B, L, H)

        # 2. Cộng Sinusoidal Positional Encoding
        embeddings = self.positional_encoding(embeddings)   # (B, L, H)

        # 3. LayerNorm + Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
