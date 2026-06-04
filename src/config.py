# =============================================================================
# config.py — Cấu hình Hyperparameters & Kiến trúc SWFT
# =============================================================================
# Tất cả hyperparameters được thiết kế dựa trên các bài báo peer-reviewed:
#   - Kiến trúc Transformer: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
#   - BERT config tham khảo: Devlin et al., NAACL 2019
#   - Factorized Embedding: Lan et al., "ALBERT", ICLR 2020
#   - Pre-LN: Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020
#   - Xavier Init: Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks", AISTATS 2010
#   - Cosine Annealing: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017
#   - Dropout as Augmentation: Gao et al., "SimCSE", EMNLP 2021
#   - Sentence Embedding Distillation: Reimers & Gurevych, EMNLP 2020
#   - AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
# =============================================================================

import os
import math
import torch


def get_device():
    """
    Tự động xác định device phù hợp.
    MPS (Apple Silicon) dùng để debug, CUDA (H100) dùng để train chính thức.
    Tham khảo: Mục 12 trong document_similarity_research.md
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# CẤU HÌNH MÔ HÌNH (SWFT — Shallow-Wide Factorized Transformer)
# =============================================================================
# Triết lý: Không phải bản thu nhỏ của BERT. Là kiến trúc thiết kế lại từ gốc.
# - Shallow (6 layers) → giảm FLOPs tuần tự, tiết kiệm compute
# - Wide (768d) → giữ sức mạnh biểu diễn ngữ nghĩa
# - Factorized Embedding (128→768) → tiết kiệm ~19M tham số (ALBERT, ICLR 2020)

MODEL_CONFIG = {
    # --- Embedding ---
    "vocab_size": 30_522,          # WordPiece vocab (BERT tokenizer) — Devlin et al., NAACL 2019
    "embedding_dim": 128,          # E = 128 (factorized, nhỏ) — Lan et al., ALBERT, ICLR 2020
    "hidden_size": 768,            # H = 768 (projection target, lớn) — giữ sức biểu diễn
    "max_seq_length": 128,         # Đủ cho sentence-level similarity

    # --- Transformer Encoder ---
    "num_layers": 6,               # Shallow: 6 layers thay vì 12 — giảm FLOPs 2x
    "num_heads": 12,               # 12 heads, mỗi head d_k = 768/12 = 64 — Vaswani et al., NeurIPS 2017
    "ffn_hidden_size": 3072,       # 4 × d_model = 3072 — tỷ lệ chuẩn Transformer
    "dropout": 0.1,                # Dropout cũng đóng vai trò Data Augmentation — SimCSE, EMNLP 2021
    "layer_norm_eps": 1e-12,       # Epsilon cho LayerNorm — theo BERT

    # --- Positional Encoding ---
    "position_encoding": "sinusoidal",  # Không thêm tham số, generalize tốt — Vaswani et al., NeurIPS 2017

    # --- Normalization ---
    "norm_type": "pre_ln",         # Pre-LN — Xiong et al., ICML 2020: ổn định gradient, giảm warmup
}

# =============================================================================
# CẤU HÌNH HUẤN LUYỆN
# =============================================================================

BUDGET_CONFIG = {
    # RunPod L40S budget plan:
    # GPU: $0.86/h, total budget: $50.
    # Network Volume 150GB for 3 days: 150 * $0.07/GB/month * 3/30 = ~$1.05.
    # Reserve $0.75 for billing jitter/setup overhead, leaving ~$48.20 GPU time.
    "total_budget_usd": float(os.environ.get("SWFT_TOTAL_BUDGET_USD", "50")),
    "gpu_usd_per_hour": float(os.environ.get("SWFT_GPU_USD_PER_HOUR", "0.86")),
    "storage_gb": float(os.environ.get("SWFT_STORAGE_GB", "150")),
    "storage_usd_per_gb_month": float(os.environ.get("SWFT_STORAGE_USD_PER_GB_MONTH", "0.07")),
    "storage_days": float(os.environ.get("SWFT_STORAGE_DAYS", "3")),
    "budget_safety_usd": float(os.environ.get("SWFT_BUDGET_SAFETY_USD", "0.75")),
}


def _storage_cost_usd() -> float:
    return (
        BUDGET_CONFIG["storage_gb"]
        * BUDGET_CONFIG["storage_usd_per_gb_month"]
        * BUDGET_CONFIG["storage_days"]
        / 30.0
    )


def _default_target_train_hours() -> float:
    available_gpu_usd = max(
        0.0,
        BUDGET_CONFIG["total_budget_usd"]
        - _storage_cost_usd()
        - BUDGET_CONFIG["budget_safety_usd"]
    )
    gpu_hours = available_gpu_usd / max(BUDGET_CONFIG["gpu_usd_per_hour"], 1e-9)
    return math.floor(gpu_hours * 10.0) / 10.0


DEFAULT_TARGET_TRAIN_HOURS = _default_target_train_hours()
DEFAULT_STAGE0_TIME_BUDGET_HOURS = min(
    40.0,
    math.floor(DEFAULT_TARGET_TRAIN_HOURS * 0.715 * 10.0) / 10.0
)

TRAIN_CONFIG = {
    # --- Optimizer: AdamW ---
    # Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
    "learning_rate": 5e-4,         # Cao hơn fine-tune (2e-5) vì train from scratch
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,

    # --- Learning Rate Schedule ---
    # Loshchilov & Hutter, "SGDR", ICLR 2017
    "scheduler": "cosine",         # Cosine Annealing — hội tụ tốt hơn linear decay
    "warmup_ratio": 0.1,           # 10% tổng steps — Vaswani et al., NeurIPS 2017

    # --- Batch Size ---
    "batch_size_debug": 8,         # Nhỏ cho MPS debug
    "batch_size_train": 64,        # Per-GPU trên H100 80GB VRAM
    # Effective optimizer batch = batch_size × accumulation.
    # Với contrastive softmax, số in-batch negatives vẫn là batch_size - 1,
    # vì gradient accumulation không ghép các similarity matrix lại với nhau.
    "gradient_accumulation_steps": 4,

    # --- Epochs ---
    "epochs_stage0": 1,            # Stage 0: supervised teacher-student distillation trên Wikipedia
    "epochs_stage1": 4,            # Stage 1: NLI fine-tune
    "epochs_stage2": 4,            # Stage 2: Similarity fine-tune

    # --- Training Budget ---
    # Default theo phương án L40S $0.86/h, tổng $50, có tính 150GB storage/3 ngày:
    # ~56 giờ tổng, Stage 0 KD ~40 giờ, phần còn lại cho Stage 1/2 và dự phòng.
    # Có thể override bằng env khi chuyển GPU hoặc cần siết chi phí.
    "target_train_hours": float(os.environ.get("SWFT_TARGET_TRAIN_HOURS", str(DEFAULT_TARGET_TRAIN_HOURS))),
    "stage0_time_budget_hours": float(os.environ.get("SWFT_STAGE0_TIME_BUDGET_HOURS", str(DEFAULT_STAGE0_TIME_BUDGET_HOURS))),
    "stage0_max_samples": int(os.environ.get("SWFT_STAGE0_MAX_SAMPLES", "0")),
    "stage0_sample_offset": int(os.environ.get("SWFT_STAGE0_SAMPLE_OFFSET", "0")),
    # Dùng giá trị nhanh/bảo thủ để cosine schedule không rơi về LR=0 quá sớm
    # nếu GPU thực tế nhanh hơn dự kiến.
    "stage0_scheduler_expected_seconds_per_batch": float(os.environ.get("SWFT_STAGE0_EXPECTED_SECONDS_PER_BATCH", "0.06")),
    "checkpoint_every_minutes": float(os.environ.get("SWFT_CHECKPOINT_EVERY_MINUTES", "20")),
    "progress_log_every_steps": int(os.environ.get("SWFT_LOG_EVERY_STEPS", "500")),

    # --- Contrastive Loss ---
    # SimCSE (EMNLP 2021): cosine similarity + temperature trong contrastive softmax
    "temperature": 0.05,

    # --- Stage 0 Teacher-Student Distillation ---
    # Reimers & Gurevych (EMNLP 2020): dùng sentence embedding teacher làm mục tiêu
    # để student học nhanh hơn với ít dữ liệu/compute hơn.
    "stage0_teacher_model": os.environ.get(
        "SWFT_STAGE0_TEACHER_MODEL",
        "sentence-transformers/all-mpnet-base-v2"
    ),
    "stage0_distillation_weight": float(os.environ.get("SWFT_STAGE0_DISTILLATION_WEIGHT", "1.0")),
    "stage0_teacher_batch_size": int(os.environ.get("SWFT_STAGE0_TEACHER_BATCH_SIZE", "64")),
    "stage0_validation_ratio": float(os.environ.get("SWFT_STAGE0_VALIDATION_RATIO", "0.02")),
    "stage0_split_seed": int(os.environ.get("SWFT_STAGE0_SPLIT_SEED", "42")),

    # --- Mixed Precision ---
    "use_amp_on_cuda": True,       # FP16 trên CUDA — tiết kiệm VRAM & tăng tốc
    "use_amp_on_mps": False,       # FP32 trên MPS — AMP chưa ổn định trên Apple Silicon

    # --- Checkpoint ---
    "checkpoint_dir": os.environ.get("SWFT_CHECKPOINT_DIR", "./checkpoints"),
    "save_every_epoch": True,      # Lưu checkpoint mỗi epoch — BẮT BUỘC cho resume
    "metrics_log_path": os.environ.get(
        "SWFT_METRICS_LOG",
        os.path.join(os.environ.get("SWFT_CHECKPOINT_DIR", "./checkpoints"), "train_metrics.jsonl")
    ),

    # --- Data Cache (Pre-tokenized, Apache Arrow) ---
    "data_cache_dir": os.environ.get("SWFT_CACHE_DIR", "./data_cache"),
}

# =============================================================================
# CẤU HÌNH DEBUG
# =============================================================================

DEBUG_CONFIG = {
    "enabled": True,               # True = debug trên MPS, False = train chính thức trên H100
    "num_samples": 3000,           # Số lượng samples debug (nhỏ để chạy nhanh)
    "eval_samples": 500,           # Số lượng samples eval debug
}

# =============================================================================
# ĐƯỜNG DẪN DỮ LIỆU
# =============================================================================

DATA_CONFIG = {
    # Dataset NLI — Bowman et al., EMNLP 2015 + Williams et al., NAACL 2018
    "nli_dataset": "stanfordnlp/snli",

    # Dataset Evaluation — Cer et al., SemEval@ACL 2017
    "stsb_dataset": "mteb/stsbenchmark-sts",

    # Dataset QQP — Wang et al., GLUE, ICLR 2019
    "qqp_dataset": "google-research-datasets/paws",   # Dùng PAWS vì đã tải

    # Dataset PAWS (Hard Negatives) — Zhang et al., NAACL 2019
    "paws_dataset": "google-research-datasets/paws",

    # Tokenizer — Devlin et al., NAACL 2019
    "tokenizer_name": "bert-base-uncased",
}
