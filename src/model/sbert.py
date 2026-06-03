# =============================================================================
# sbert.py — Siamese Bi-Encoder với Mean Pooling (Tự implement từ đầu)
# =============================================================================
# Implement 100% bằng PyTorch thuần.
#
# Bài báo tham khảo:
#   [1] Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese
#       BERT-Networks", EMNLP 2019
#       → Kiến trúc Siamese Bi-Encoder: 2 câu encode ĐỘC LẬP, chia sẻ trọng số
#       → Mean Pooling (tốt nhất theo paper): trung bình cộng có attention mask
#       → So sánh 10,000 tài liệu: ~5 giây (vs 65 giờ cross-encoder)
#
#   [2] Vaswani et al., NeurIPS 2017 — Transformer Encoder backbone
#   [3] Lan et al., ICLR 2020 — Factorized Embedding
#   [4] Xiong et al., ICML 2020 — Pre-LN
# =============================================================================

import torch
import torch.nn as nn
from model.embedding import TransformerEmbedding
from model.transformer import TransformerEncoder


class MeanPooling(nn.Module):
    """
    Mean Pooling (Attention-Mask Aware) — Reimers & Gurevych, EMNLP 2019.

    Thay vì lấy vector [CLS] (chỉ 1 token đại diện cả câu → thất thoát thông tin),
    Mean Pooling tính TRUNG BÌNH CỘNG tất cả token embeddings, nhưng BỎ QUA padding tokens.

    Công thức:
        v = Σ(h_i × mask_i) / Σ(mask_i)

    Trong đó:
        h_i    = hidden state của token thứ i
        mask_i = 1 nếu token thật, 0 nếu padding
        → Đảm bảo padding tokens không ảnh hưởng đến sentence embedding

    Kết quả thí nghiệm trong SBERT paper (Table 1):
        Mean Pooling > [CLS] Token > Max Pooling
    """

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states:  (batch_size, seq_len, hidden_size) — output của Transformer
            attention_mask: (batch_size, seq_len) — 1 cho token thật, 0 cho padding

        Returns:
            (batch_size, hidden_size) — 1 vector đại diện cho cả câu
        """
        # Mở rộng mask để nhân element-wise với hidden states
        # (B, L) → (B, L, 1) — broadcast theo hidden_size dimension
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)

        # Nhân hidden states với mask → zero out padding tokens
        # Sau đó sum theo seq_len dimension
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)  # (B, H)

        # Tính tổng mask (= số token thật trong mỗi câu)
        # clamp(min=1e-9) để tránh chia cho 0
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B, 1)

        # Trung bình cộng = tổng embeddings / số token thật
        mean_embeddings = sum_embeddings / sum_mask  # (B, H)

        return mean_embeddings


class SWFTModel(nn.Module):
    """
    SWFT (Shallow-Wide Factorized Transformer) — Siamese Bi-Encoder.

    Kiến trúc hoàn chỉnh:
        Token IDs → Factorized Embedding → Transformer Encoder → Mean Pooling → Sentence Vector

    Đây là mô hình Siamese (Reimers & Gurevych, EMNLP 2019):
        - Câu A và Câu B đi qua CÙNG MỘT encoder (chia sẻ trọng số)
        - Mỗi câu cho ra 1 vector embedding ĐỘC LẬP
        - So sánh bằng Cosine Similarity
        - Ưu điểm: pre-compute embeddings cho corpus → tìm kiếm siêu nhanh

    Tổng tham số với config mặc định hiện tại:
        Factorized Embedding: 30,522×128 + 128×768 ≈ 4.0M
        6 Encoder Blocks: 6 × ~7.09M ≈ 42.5M
        LayerNorm + Final: ≈ 0.01M
        ───────────────────────────────
        Tổng: ≈ 46.5M tham số (vs BERT-base ~110M = giảm ~2.4×)
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 max_seq_length: int, num_layers: int, num_heads: int,
                 ffn_hidden_size: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-12):
        super().__init__()

        # ===== Lớp 1: Embedding =====
        # Factorized Embedding (ALBERT, ICLR 2020) + Sinusoidal PE (Vaswani, NeurIPS 2017)
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,     # E = 128 (factorized)
            hidden_size=hidden_size,         # H = 768 (target)
            max_seq_length=max_seq_length,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout
        )

        # ===== Lớp 2: Transformer Encoder =====
        # Pre-LN (Xiong et al., ICML 2020) × 6 layers
        self.encoder = TransformerEncoder(
            num_layers=num_layers,           # 6 layers (shallow)
            d_model=hidden_size,             # 768 (wide)
            num_heads=num_heads,             # 12 heads
            d_ff=ffn_hidden_size,            # 3072 = 4 × 768
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        # ===== Lớp 3: Mean Pooling =====
        # Reimers & Gurevych, EMNLP 2019
        self.pooling = MeanPooling()

        # Lưu config
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Token IDs → Sentence Embedding vector.

        Args:
            input_ids:      (batch_size, seq_len) — Token IDs
            attention_mask: (batch_size, seq_len) — 1=token thật, 0=padding

        Returns:
            (batch_size, hidden_size) — Sentence embedding v ∈ ℝ^768
        """
        # 1. Factorized Embedding + Sinusoidal PE
        embeddings = self.embedding(input_ids)  # (B, L, H=768)

        # 2. Transformer Encoder (6 layers, Pre-LN)
        hidden_states = self.encoder(embeddings, attention_mask)  # (B, L, H=768)

        # 3. Mean Pooling → 1 vector cho cả câu
        sentence_embedding = self.pooling(hidden_states, attention_mask)  # (B, H=768)

        return sentence_embedding

    def count_parameters(self) -> int:
        """Đếm tổng số tham số học được."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_swft_model(config: dict) -> SWFTModel:
    """
    Factory function tạo SWFT model từ config dict.

    Args:
        config: MODEL_CONFIG từ config.py

    Returns:
        SWFTModel instance
    """
    model = SWFTModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        max_seq_length=config["max_seq_length"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ffn_hidden_size=config["ffn_hidden_size"],
        dropout=config["dropout"],
        layer_norm_eps=config["layer_norm_eps"]
    )
    return model
