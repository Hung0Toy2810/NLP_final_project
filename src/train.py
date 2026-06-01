# =============================================================================
# train.py — Vòng lặp huấn luyện PyTorch thuần (Custom Training Loop)
# =============================================================================
# KHÔNG dùng model.fit(), KHÔNG dùng HuggingFace Trainer.
# Mọi bước đều minh bạch: zero_grad → forward → loss → backward → step.
#
# Curriculum Learning Pipeline (Bengio et al., ICML 2009):
#   Stage 0: Unsupervised SimCSE trên Wikipedia (Gao et al., EMNLP 2021)
#   Stage 1: NLI SoftmaxLoss (Reimers & Gurevych, EMNLP 2019)
#   Stage 2: MNR Loss + Hard Negatives trên PAWS (Zhang et al., NAACL 2019)
#
# Bài báo tham khảo:
#   [1] Reimers & Gurevych, EMNLP 2019 — Sentence-BERT
#   [2] Bengio et al., ICML 2009 — Curriculum Learning: dễ → khó
#   [3] Loshchilov & Hutter, ICLR 2017 — Cosine Annealing LR Schedule
#   [4] Loshchilov & Hutter, ICLR 2019 — AdamW Optimizer
#   [5] Vaswani et al., NeurIPS 2017 — Linear Warmup
#   [6] Gao et al., EMNLP 2021 — SimCSE: Dropout as Data Augmentation
# =============================================================================

import os
import sys
import logging
import math
import torch
import torch.nn as nn
from torch.optim import AdamW

# Thêm src vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, TRAIN_CONFIG, DEBUG_CONFIG, DATA_CONFIG, get_device
from model.sbert import create_swft_model
from losses import SoftmaxLoss, MultipleNegativesRankingLoss
from dataset import (
    get_tokenizer, WikipediaSimCSEDataset, NLIDataset, SimilarityDataset,
    STSBDataset, create_dataloader
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# COSINE ANNEALING WITH WARMUP
# =============================================================================
# Loshchilov & Hutter, "SGDR", ICLR 2017 + Vaswani et al., NeurIPS 2017

class CosineAnnealingWithWarmup:
    """
    Learning Rate Schedule: Linear Warmup + Cosine Annealing.

    Phase 1 — Linear Warmup (Vaswani et al., NeurIPS 2017):
        LR tăng tuyến tính từ 0 đến lr_max trong warmup_steps đầu tiên.
        Giúp mô hình không bị "sốc" khi gradient lớn ở đầu training.

    Phase 2 — Cosine Annealing (Loshchilov & Hutter, ICLR 2017):
        LR giảm theo hàm cosine từ lr_max → 0.
        Giảm mượt hơn linear decay, cho phép model tinh chỉnh cuối training.

    Công thức:
        if step < warmup_steps:
            lr = lr_max × (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = lr_max × 0.5 × (1 + cos(π × progress))
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear Warmup
            lr = self.base_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine Annealing
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# =============================================================================
# CHECKPOINT: SAVE & LOAD (Resume Training)
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    """
    Lưu checkpoint để resume training.
    BẮT BUỘC khi train trên cloud (session có thể bị ngắt bất cứ lúc nào).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_step': scheduler.current_step,
        'epoch': epoch,
        'step': step,
        'loss': loss,
    }
    torch.save(checkpoint, path)
    logger.info(f" Checkpoint saved: {path} (epoch={epoch}, step={step}, loss={loss:.4f})")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """
    Load checkpoint để resume training.
    Returns: (epoch, step) nếu tìm thấy checkpoint, (0, 0) nếu không.
    """
    if not os.path.exists(path):
        logger.info("Không tìm thấy checkpoint. Bắt đầu training từ đầu.")
        return 0, 0

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.current_step = checkpoint['scheduler_step']

    epoch = checkpoint['epoch']
    step = checkpoint['step']
    loss = checkpoint['loss']
    logger.info(f" Checkpoint loaded: {path} (epoch={epoch}, step={step}, loss={loss:.4f})")

    return epoch, step


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_stsb(model, eval_dataloader, device):
    """
    Đánh giá trên STS-B bằng Spearman Correlation.
    Cer et al., SemEval@ACL 2017.
    """
    from scipy.stats import spearmanr

    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            ids_b = batch['input_ids_b'].to(device)
            mask_b = batch['attention_mask_b'].to(device)

            emb_a = model(ids_a, mask_a)
            emb_b = model(ids_b, mask_b)

            # Cosine similarity
            cos_sim = nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)
            all_scores.extend(cos_sim.cpu().tolist())
            all_labels.extend(batch['score'].tolist())

    spearman_corr, _ = spearmanr(all_scores, all_labels)
    model.train()
    return spearman_corr


def _create_eval_loader(device, debug, cache_dir):
    """Helper: tạo eval dataloader cho STS-B."""
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    batch_size = TRAIN_CONFIG["batch_size_debug"] if debug else TRAIN_CONFIG["batch_size_train"]
    num_workers = 0 if device.type != "cuda" else 4

    stsb_dataset = STSBDataset(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        max_length=MODEL_CONFIG["max_seq_length"],
        split="validation",
        debug=debug,
        num_debug_samples=DEBUG_CONFIG["eval_samples"]
    )
    return create_dataloader(stsb_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             prefetch_factor=2 if num_workers > 0 else None)


# =============================================================================
# GIAI ĐOẠN 0: UNSUPERVISED SimCSE (Wikipedia)
# =============================================================================
# Gao et al., "SimCSE: Simple Contrastive Learning of Sentence Embeddings",
# EMNLP 2021.
#
# Ý tưởng cốt lõi (trích paper, Section 3.1):
#   "We simply feed the same input sentence to the encoder TWICE and obtain
#    two different embeddings as 'positive pairs', by applying independently
#    sampled dropout masks."
#
# Giải thích chi tiết:
#   1. Mỗi câu x đi vào mô hình 2 lần:
#      h₁ = model(x)  ← Dropout mask lần 1 (ngẫu nhiên)
#      h₂ = model(x)  ← Dropout mask lần 2 (ngẫu nhiên KHÁC)
#   2. h₁ ≠ h₂ mặc dù input giống nhau (do Dropout khác nhau)
#   3. MNR Loss ép h₁ và h₂ lại gần nhau, đẩy các câu khác trong batch ra xa
#   4. Kết quả: mô hình học được rằng "nội dung giống → embedding giống"

def train_stage0_simcse(model, device, debug=True):
    """
    Giai đoạn 0: Unsupervised SimCSE — Gao et al., EMNLP 2021.
    Curriculum Learning Stage 0 (dễ nhất) — Bengio et al., ICML 2009.

    Mục tiêu: Học biểu diễn ngôn ngữ cơ bản từ text thô Wikipedia
    Loss: MNR Loss (cùng 1 câu encode 2 lần, Dropout tạo positive pair)
    """
    logger.info("=" * 60)
    logger.info("GIAI ĐOẠN 0: UNSUPERVISED SimCSE (Wikipedia)")
    logger.info("Paper: Gao et al., EMNLP 2021 + Bengio et al., ICML 2009")
    logger.info("Dropout as Data Augmentation — không cần nhãn!")
    logger.info("=" * 60)

    cache_dir = TRAIN_CONFIG.get("data_cache_dir")
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    batch_size = TRAIN_CONFIG["batch_size_debug"] if debug else TRAIN_CONFIG["batch_size_train"]
    num_workers = 0 if device.type != "cuda" else 8  # Wikipedia lớn → cần nhiều workers

    # Dataset
    wiki_dataset = WikipediaSimCSEDataset(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        max_length=MODEL_CONFIG["max_seq_length"],
        debug=debug,
        num_debug_samples=DEBUG_CONFIG["num_samples"]
    )
    train_loader = create_dataloader(wiki_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     prefetch_factor=2 if num_workers > 0 else None)

    # Evaluation
    eval_loader = _create_eval_loader(device, debug, cache_dir)

    # Loss — MNR Loss (SimCSE sử dụng InfoNCE / MNR Loss)
    criterion = MultipleNegativesRankingLoss(temperature=TRAIN_CONFIG["temperature"])

    # Optimizer — AdamW, Loshchilov & Hutter, ICLR 2019
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(TRAIN_CONFIG["adam_beta1"], TRAIN_CONFIG["adam_beta2"]),
        eps=TRAIN_CONFIG["adam_epsilon"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )

    # LR Schedule
    epochs = TRAIN_CONFIG["epochs_stage0"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(TRAIN_CONFIG["warmup_ratio"] * total_steps)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # Mixed Precision
    use_amp = TRAIN_CONFIG["use_amp_on_cuda"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Resume
    checkpoint_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage0_latest.pt")
    start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)

    logger.info(f"Wikipedia sentences: {len(wiki_dataset):,}")
    logger.info(f"Epochs: {epochs}, Total steps: {total_steps}, Warmup: {warmup_steps}")
    logger.info(f"Batch size: {batch_size}, AMP: {use_amp}, Device: {device}")

    # ===== TRAINING LOOP =====
    model.train()
    global_step = start_step

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < (start_step % len(train_loader)):
                continue

            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # ===== CỐT LÕI CỦA SimCSE =====
                # Cùng 1 input, encode 2 lần → Dropout khác nhau → 2 embeddings khác nhau
                # (Gao et al., EMNLP 2021, Section 3.1)
                model.train()  # Đảm bảo Dropout BẬT
                embedding_1 = model(ids, mask)  # h₁ = f(x, z₁) với z₁ = dropout mask 1
                embedding_2 = model(ids, mask)  # h₂ = f(x, z₂) với z₂ = dropout mask 2

                # MNR Loss: ép h₁ ≈ h₂ (cùng câu), đẩy h_i ≠ h_j (câu khác)
                loss = criterion(embedding_1, embedding_2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                avg_loss = epoch_loss / num_batches
                lr = scheduler.get_lr()
                logger.info(f"[Stage0-SimCSE] Epoch {epoch+1}/{epochs} | Step {global_step} | "
                           f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        # Evaluate
        spearman = evaluate_stsb(model, eval_loader, device)
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"[Stage0-SimCSE] Epoch {epoch+1}/{epochs} DONE | "
                   f"Avg Loss: {avg_loss:.4f} | STS-B Spearman: {spearman:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_loss,
                       checkpoint_path)

    final_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage0_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f" Giai đoạn 0 hoàn thành. Model saved: {final_path}")


# =============================================================================
# GIAI ĐOẠN 1: NLI TRAINING
# =============================================================================

def train_stage1_nli(model, device, debug=True):
    """
    Giai đoạn 1: NLI Fine-tune — Reimers & Gurevych, EMNLP 2019.
    Curriculum Learning Stage 1 (dễ) — Bengio et al., ICML 2009.

    Mục tiêu: Học phân biệt ngữ nghĩa tổng quát (Entailment vs Contradiction vs Neutral)
    Loss: SoftmaxLoss — concat(u, v, |u-v|) → Linear → CrossEntropy
    """
    logger.info("=" * 60)
    logger.info("GIAI ĐOẠN 1: NLI TRAINING (Curriculum Learning — Easy)")
    logger.info("Paper: Reimers & Gurevych, EMNLP 2019 + Bengio et al., ICML 2009")
    logger.info("=" * 60)

    cache_dir = TRAIN_CONFIG.get("data_cache_dir")
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    batch_size = TRAIN_CONFIG["batch_size_debug"] if debug else TRAIN_CONFIG["batch_size_train"]
    num_workers = 0 if device.type != "cuda" else 4

    # Dataset
    nli_dataset = NLIDataset(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        max_length=MODEL_CONFIG["max_seq_length"],
        debug=debug,
        num_debug_samples=DEBUG_CONFIG["num_samples"]
    )
    train_loader = create_dataloader(nli_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     prefetch_factor=2 if num_workers > 0 else None)

    # Evaluation
    eval_loader = _create_eval_loader(device, debug, cache_dir)

    # Loss — SoftmaxLoss, Reimers & Gurevych, EMNLP 2019
    criterion = SoftmaxLoss(hidden_size=MODEL_CONFIG["hidden_size"], num_labels=3).to(device)

    # Optimizer — AdamW, Loshchilov & Hutter, ICLR 2019
    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(TRAIN_CONFIG["adam_beta1"], TRAIN_CONFIG["adam_beta2"]),
        eps=TRAIN_CONFIG["adam_epsilon"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )

    # LR Schedule — Cosine Annealing with Warmup
    epochs = TRAIN_CONFIG["epochs_stage1"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(TRAIN_CONFIG["warmup_ratio"] * total_steps)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # Mixed Precision — FP16 trên CUDA, FP32 trên MPS
    use_amp = TRAIN_CONFIG["use_amp_on_cuda"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Resume from checkpoint
    checkpoint_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage1_latest.pt")
    start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)

    logger.info(f"Epochs: {epochs}, Total steps: {total_steps}, Warmup: {warmup_steps}")
    logger.info(f"Batch size: {batch_size}, AMP: {use_amp}, Device: {device}")

    # ===== TRAINING LOOP =====
    model.train()
    global_step = start_step

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < (start_step % len(train_loader)):
                continue  # Skip batches already processed

            # Chuyển data lên device
            ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            ids_b = batch['input_ids_b'].to(device)
            mask_b = batch['attention_mask_b'].to(device)
            labels = batch['label'].to(device)

            # ===== Forward Pass =====
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Encode 2 câu qua Siamese Encoder (chia sẻ trọng số)
                embedding_a = model(ids_a, mask_a)  # (B, H)
                embedding_b = model(ids_b, mask_b)  # (B, H)

                # SoftmaxLoss: concat(u, v, |u-v|) → classifier → CE
                loss = criterion(embedding_a, embedding_b, labels)

            # ===== Backward Pass =====
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # LR Schedule step
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Log every 50 steps
            if global_step % 50 == 0:
                avg_loss = epoch_loss / num_batches
                lr = scheduler.get_lr()
                logger.info(f"[Stage1] Epoch {epoch+1}/{epochs} | Step {global_step} | "
                           f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        # Evaluate after each epoch
        spearman = evaluate_stsb(model, eval_loader, device)
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"[Stage1] Epoch {epoch+1}/{epochs} DONE | "
                   f"Avg Loss: {avg_loss:.4f} | STS-B Spearman: {spearman:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_loss,
                       checkpoint_path)

    # Save final model
    final_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage1_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f" Giai đoạn 1 hoàn thành. Model saved: {final_path}")


# =============================================================================
# GIAI ĐOẠN 2: SIMILARITY TRAINING (Contrastive Learning)
# =============================================================================

def train_stage2_similarity(model, device, debug=True, include_hard_negatives=False):
    """
    Giai đoạn 2: Similarity Fine-tune — Reimers & Gurevych, EMNLP 2019.
    Curriculum Learning Stage 2 (khó hơn) — Bengio et al., ICML 2009.

    Mục tiêu: Tối ưu cosine similarity giữa các cặp câu tương đồng
    Loss: MNR Loss (Multiple Negatives Ranking) — Henderson et al., 2017
    Hard Negatives: PAWS — Zhang et al., NAACL 2019
    """
    stage_name = "Stage2+HN" if include_hard_negatives else "Stage2"
    logger.info("=" * 60)
    logger.info(f"GIAI ĐOẠN 2: SIMILARITY TRAINING ({stage_name})")
    logger.info("Paper: Henderson et al., 2017 (MNR) + Zhang et al., NAACL 2019 (PAWS)")
    logger.info("=" * 60)

    cache_dir = TRAIN_CONFIG.get("data_cache_dir")
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    batch_size = TRAIN_CONFIG["batch_size_debug"] if debug else TRAIN_CONFIG["batch_size_train"]
    num_workers = 0 if device.type != "cuda" else 4

    # Dataset
    sim_dataset = SimilarityDataset(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        max_length=MODEL_CONFIG["max_seq_length"],
        debug=debug,
        num_debug_samples=DEBUG_CONFIG["num_samples"],
        include_hard_negatives=include_hard_negatives
    )
    train_loader = create_dataloader(sim_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     prefetch_factor=2 if num_workers > 0 else None)

    # Evaluation
    eval_loader = _create_eval_loader(device, debug, cache_dir)

    # Loss — MNR Loss, Henderson et al., 2017
    criterion = MultipleNegativesRankingLoss(temperature=TRAIN_CONFIG["temperature"])

    # Optimizer — AdamW
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(TRAIN_CONFIG["adam_beta1"], TRAIN_CONFIG["adam_beta2"]),
        eps=TRAIN_CONFIG["adam_epsilon"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )

    # LR Schedule
    epochs = TRAIN_CONFIG["epochs_stage2"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(TRAIN_CONFIG["warmup_ratio"] * total_steps)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # Mixed Precision
    use_amp = TRAIN_CONFIG["use_amp_on_cuda"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Resume from checkpoint
    ckpt_name = "stage2_hn_latest.pt" if include_hard_negatives else "stage2_latest.pt"
    checkpoint_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], ckpt_name)
    start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)

    logger.info(f"Epochs: {epochs}, Total steps: {total_steps}, Warmup: {warmup_steps}")
    logger.info(f"Batch size: {batch_size}, Temperature: {TRAIN_CONFIG['temperature']}")

    # ===== TRAINING LOOP =====
    model.train()
    global_step = start_step

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < (start_step % len(train_loader)):
                continue

            ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            ids_b = batch['input_ids_b'].to(device)
            mask_b = batch['attention_mask_b'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Encode anchor và positive
                embedding_a = model(ids_a, mask_a)  # (B, H)
                embedding_b = model(ids_b, mask_b)  # (B, H)

                # MNR Loss: cosine similarity matrix + cross-entropy
                loss = criterion(embedding_a, embedding_b)

                # Nếu có hard negatives, thêm loss cho (anchor, negative)
                if include_hard_negatives and 'input_ids_neg' in batch:
                    ids_neg = batch['input_ids_neg'].to(device)
                    mask_neg = batch['attention_mask_neg'].to(device)

                    # Encode negative
                    embedding_neg = model(ids_neg, mask_neg)

                    # Ghép negative vào positive pool → MNR Loss khó hơn
                    all_positives = torch.cat([embedding_b, embedding_neg], dim=0)
                    loss_hn = criterion(embedding_a, all_positives[:len(embedding_a)])
                    loss = (loss + loss_hn) / 2.0

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = epoch_loss / num_batches
                lr = scheduler.get_lr()
                logger.info(f"[{stage_name}] Epoch {epoch+1}/{epochs} | Step {global_step} | "
                           f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        spearman = evaluate_stsb(model, eval_loader, device)
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"[{stage_name}] Epoch {epoch+1}/{epochs} DONE | "
                   f"Avg Loss: {avg_loss:.4f} | STS-B Spearman: {spearman:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_loss,
                       checkpoint_path)

    final_name = "stage2_hn_final.pt" if include_hard_negatives else "stage2_final.pt"
    final_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], final_name)
    torch.save(model.state_dict(), final_path)
    logger.info(f" Giai đoạn 2 ({stage_name}) hoàn thành. Model saved: {final_path}")


# =============================================================================
# MAIN — Pipeline hoàn chỉnh
# =============================================================================

def main():
    device = get_device()
    debug = DEBUG_CONFIG["enabled"]

    logger.info("=" * 60)
    logger.info("SWFT — Shallow-Wide Factorized Transformer")
    logger.info("Train from scratch — Pure PyTorch")
    logger.info(f"Device: {device} | Debug: {debug}")
    logger.info("=" * 60)

    # Tạo model
    model = create_swft_model(MODEL_CONFIG).to(device)
    total_params = model.count_parameters()
    logger.info(f"Tổng tham số: {total_params:,} (~{total_params/1e6:.1f}M)")

    # ===== Curriculum Learning (Bengio et al., ICML 2009) =====
    # Stage 0: Unsupervised SimCSE (dễ nhất — chỉ cần text thô)
    # Stage 1: NLI (dễ — phân loại 3 classes)
    # Stage 2: Similarity + Hard Negatives (khó nhất — contrastive)

    # Giai đoạn 0: Unsupervised SimCSE trên Wikipedia
    train_stage0_simcse(model, device, debug=debug)

    # Giai đoạn 1: NLI
    train_stage1_nli(model, device, debug=debug)

    # Giai đoạn 2: Similarity (MNR Loss) trên PAWS
    train_stage2_similarity(model, device, debug=debug, include_hard_negatives=False)

    # Giai đoạn 2+: Similarity + Hard Negatives (PAWS adversarial)
    train_stage2_similarity(model, device, debug=debug, include_hard_negatives=True)

    logger.info("🎉 HOÀN THÀNH TOÀN BỘ PIPELINE TRAINING!")


if __name__ == "__main__":
    main()
