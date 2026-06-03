# =============================================================================
# losses.py — SoftmaxLoss + Contrastive Loss + Teacher Distillation
# =============================================================================
# Implement 100% bằng PyTorch thuần (torch.nn.functional).
#
# Bài báo tham khảo:
#   [1] Reimers & Gurevych, "Sentence-BERT", EMNLP 2019
#       → SoftmaxLoss cho NLI: concat(u, v, |u-v|) → Linear → Softmax → CE
#       → 3 classes: Entailment, Contradiction, Neutral
#
#   [2] Gao et al., "SimCSE", EMNLP 2021
#       → Cosine similarity + temperature scaling cho contrastive learning
#       → Unsupervised: cùng input encode 2 lần bằng dropout
#       → Supervised: entailment positives, contradiction hard negatives
#
#   [3] Reimers & Gurevych, "Making Monolingual Sentence Embeddings
#       Multilingual using Knowledge Distillation", EMNLP 2020
#       → Student sentence encoder học bắt chước embedding space của teacher
#
#   [4] TinyBERT, Findings EMNLP 2020 + MiniLM, NeurIPS 2020
#       → Distillation là hướng hợp lệ để nén/tăng tốc Transformer student
#
#   [5] Glorot & Bengio, AISTATS 2010
#       → Xavier Init cho Linear layers trong SoftmaxLoss
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SoftmaxLoss(nn.Module):
    """
    SoftmaxLoss cho NLI — Reimers & Gurevych, EMNLP 2019, Section 3.

    Dùng cho Giai đoạn 1 (NLI Fine-tune):
        Input: 2 câu (premise, hypothesis)
        Output: 3 classes (Entailment, Contradiction, Neutral)

    Kiến trúc:
        u = encode(sentence_A)                    # (B, H)
        v = encode(sentence_B)                    # (B, H)
        features = concat(u, v, |u - v|)          # (B, 3H)
        logits = Linear(3H → num_labels)          # (B, 3)
        loss = CrossEntropy(logits, labels)

    Tại sao dùng |u - v| (element-wise absolute difference)?
        → Capture "mức độ khác biệt" giữa 2 câu ở mỗi dimension
        → Kết hợp với u và v cho model cả thông tin "giống" và "khác"
        → SBERT paper chứng minh concat(u, v, |u-v|) tốt hơn chỉ concat(u, v)
    """

    def __init__(self, hidden_size: int, num_labels: int = 3):
        super().__init__()

        # Input: concat(u, v, |u-v|) → 3 × hidden_size
        self.classifier = nn.Linear(3 * hidden_size, num_labels)

        # Xavier Init — Glorot & Bengio, AISTATS 2010
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding_a: (batch_size, hidden_size) — sentence embedding của câu A
            embedding_b: (batch_size, hidden_size) — sentence embedding của câu B
            labels:      (batch_size,) — nhãn NLI (0=Entailment, 1=Neutral, 2=Contradiction)

        Returns:
            loss: scalar — Cross-Entropy Loss
        """
        # Bước 1: Tạo feature vector = concat(u, v, |u-v|)
        diff = torch.abs(embedding_a - embedding_b)  # |u - v|
        features = torch.cat([embedding_a, embedding_b, diff], dim=-1)  # (B, 3H)

        # Bước 2: Linear projection → logits
        logits = self.classifier(features)  # (B, num_labels=3)

        # Bước 3: Cross-Entropy Loss
        loss = F.cross_entropy(logits, labels)

        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """
    Contrastive softmax loss với in-batch negatives.

    Tên class giữ là MultipleNegativesRankingLoss để phản ánh cách gọi phổ biến
    trong sentence embedding libraries, nhưng công thức ở đây bám theo SimCSE
    (Gao et al., EMNLP 2021): cosine similarity + temperature + cross entropy.

    Ý tưởng cốt lõi: In-Batch Negatives
        Cho batch gồm N cặp (anchor, positive): {(a₁,p₁), (a₂,p₂), ..., (aₙ,pₙ)}

        Với mỗi anchor aᵢ:
            - Positive = pᵢ (câu tương đồng đúng)
            - Negatives = {p₁, p₂, ..., pᵢ₋₁, pᵢ₊₁, ..., pₙ} (tất cả positive khác trong batch)

        → Không cần label negative tường minh!
        → Batch size lớn = nhiều negatives hơn = model mạnh hơn

    Công thức Loss (cho mỗi anchor aᵢ):
        L = -log( exp(sim(aᵢ, pᵢ) / τ) / Σⱼ exp(sim(aᵢ, pⱼ) / τ) )

    Trong đó:
        sim(a, p) = cosine_similarity(a, p)
        τ = temperature (thường = 0.05)

    Implement bằng ma trận:
        1. Tính similarity matrix: S = A × Pᵀ           (N × N)
        2. Chia cho temperature:   S = S / τ
        3. Labels = đường chéo chính:  labels = [0, 1, 2, ..., N-1]
           (vì positive của aᵢ nằm ở cột i)
        4. Loss = CrossEntropy(S, labels)
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            anchor_embeddings:   (batch_size, hidden_size) — embeddings của anchors
            positive_embeddings: (batch_size, hidden_size) — embeddings của positives
            negative_embeddings: optional hard negatives, shape:
                                 (batch_size, hidden_size) hoặc
                                 (batch_size, num_negatives, hidden_size)

        Returns:
            loss: scalar — contrastive loss

        Giải thích từng bước toán học:
        """
        if anchor_embeddings.size(0) != positive_embeddings.size(0):
            raise ValueError(
                "anchor_embeddings và positive_embeddings phải có cùng batch_size "
                f"({anchor_embeddings.size(0)} != {positive_embeddings.size(0)})"
            )

        # ===== Bước 1: Normalize embeddings (để cosine similarity = dot product) =====
        # ||a|| = 1, ||p|| = 1 → cos(a, p) = a · p
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1)    # (B, H)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=-1)  # (B, H)

        candidates = positive_norm

        # Hard negatives đi vào denominator của cùng softmax.
        # Positive của anchor_i vẫn nằm ở cột i; negative columns không bao giờ là label.
        if negative_embeddings is not None:
            negative_norm = F.normalize(negative_embeddings, p=2, dim=-1)
            if negative_norm.dim() == 3:
                negative_norm = negative_norm.reshape(-1, negative_norm.size(-1))
            elif negative_norm.dim() != 2:
                raise ValueError(
                    "negative_embeddings phải có shape (B, H) hoặc (B, K, H), "
                    f"nhận được {tuple(negative_embeddings.shape)}"
                )
            candidates = torch.cat([positive_norm, negative_norm], dim=0)

        # ===== Bước 2: Tính similarity matrix =====
        # Không có hard negatives: S shape (B, B).
        # Có hard negatives: S shape (B, B + num_negative_candidates).
        similarity_matrix = torch.matmul(anchor_norm, candidates.t())

        # ===== Bước 3: Scale bằng temperature =====
        # Temperature nhỏ (0.05) → phân phối "sắc nét" hơn
        # → model bị phạt nặng hơn khi nhầm negative thành positive
        similarity_matrix = similarity_matrix / self.temperature  # (B, num_candidates)

        # ===== Bước 4: Tạo labels =====
        # Positive của anchor_i nằm ở cột i → labels = [0, 1, 2, ..., B-1]
        # Đây chính là đường chéo chính (diagonal) của similarity matrix
        batch_size = anchor_embeddings.size(0)
        labels = torch.arange(batch_size, device=anchor_embeddings.device)

        # ===== Bước 5: Cross-Entropy Loss =====
        # CE loss tự động áp dụng log-softmax rồi negative log likelihood
        # Tương đương: L = -log(exp(S[i][i]) / Σⱼ exp(S[i][j]))
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class Stage0TeacherDistillationLoss(nn.Module):
    """
    Direct teacher-student distillation cho Stage 0 sentence embeddings.

    Teacher được xem như ground truth embedding space:
        - Teacher chạy eval/no_grad ở training loop.
        - Student là nhánh duy nhất nhận gradient.
        - Cả student và teacher đều được L2-normalize.
        - Loss = 1 - cosine(student, teacher).

    Loss này cố ý yêu cầu teacher_dim == student_dim. Nếu muốn student 768d học
    trực tiếp vector teacher thì teacher cũng phải xuất 768d, ví dụ
    sentence-transformers/all-mpnet-base-v2.

    Cơ sở:
        - Reimers & Gurevych, EMNLP 2020: student sentence embedding model học
          mimic embedding space từ teacher.
    """

    def __init__(self):
        super().__init__()

    def forward(self, student_embeddings: torch.Tensor,
                teacher_embeddings: torch.Tensor) -> torch.Tensor:
        if student_embeddings.size(0) != teacher_embeddings.size(0):
            raise ValueError(
                "student_embeddings và teacher_embeddings phải có cùng batch_size "
                f"({student_embeddings.size(0)} != {teacher_embeddings.size(0)})"
            )
        if student_embeddings.size(-1) != teacher_embeddings.size(-1):
            raise ValueError(
                "Direct Stage 0 distillation yêu cầu student_dim == teacher_dim "
                f"({student_embeddings.size(-1)} != {teacher_embeddings.size(-1)}). "
                "Hãy dùng teacher 768d, ví dụ sentence-transformers/all-mpnet-base-v2."
            )

        student = F.normalize(student_embeddings.float(), p=2, dim=-1)
        teacher = F.normalize(teacher_embeddings.detach().float(), p=2, dim=-1)
        cosine = (student * teacher).sum(dim=-1)
        return 1.0 - cosine.mean()
