# =============================================================================
# train.py — Vòng lặp huấn luyện PyTorch thuần (Custom Training Loop)
# =============================================================================
# KHÔNG dùng model.fit(), KHÔNG dùng HuggingFace Trainer.
# Mọi bước đều minh bạch: zero_grad → forward → loss → backward → step.
#
# Curriculum Learning Pipeline (Bengio et al., ICML 2009):
#   Stage 0: Supervised teacher-student distillation trên Wikipedia
#   Stage 1: NLI SoftmaxLoss (Reimers & Gurevych, EMNLP 2019)
#   Stage 2: Contrastive softmax + Hard Negatives trên PAWS (Zhang et al., NAACL 2019)
#
# Bài báo tham khảo:
#   [1] Reimers & Gurevych, EMNLP 2019 — Sentence-BERT
#   [2] Bengio et al., ICML 2009 — Curriculum Learning: dễ → khó
#   [3] Loshchilov & Hutter, ICLR 2017 — Cosine Annealing LR Schedule
#   [4] Loshchilov & Hutter, ICLR 2019 — AdamW Optimizer
#   [5] Vaswani et al., NeurIPS 2017 — Linear Warmup
#   [6] Gao et al., EMNLP 2021 — SimCSE: Dropout as Data Augmentation
#   [7] Reimers & Gurevych, EMNLP 2020 — Sentence Embedding Distillation
# =============================================================================

import os
import sys
import logging
import math
import time
import shutil
import torch
import torch.nn as nn
from typing import Optional
from torch.optim import AdamW
from torch.amp import GradScaler, autocast  # pyright: ignore[reportPrivateImportUsage]

# Thêm src vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, TRAIN_CONFIG, DEBUG_CONFIG, DATA_CONFIG, get_device
from model.sbert import create_swft_model
from losses import (
    SoftmaxLoss, MultipleNegativesRankingLoss, Stage0TeacherDistillationLoss
)
from dataset import (
    get_tokenizer, WikipediaDistillationDataset, NLIDataset, SimilarityDataset,
    STSBDataset, create_dataloader
)
from training_log import init_training_logger, log_event

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
            progress = min(1.0, max(0.0, progress))
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# =============================================================================
# CHECKPOINT: SAVE & LOAD (Resume Training)
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path,
                    extra_modules: Optional[dict] = None,
                    stage_name: Optional[str] = None):
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
    if extra_modules:
        checkpoint['extra_module_state_dicts'] = {
            name: module.state_dict()
            for name, module in extra_modules.items()
        }
    torch.save(checkpoint, path)
    logger.info(f" Checkpoint saved: {path} (epoch={epoch}, step={step}, loss={loss:.4f})")
    log_event(
        "checkpoint",
        stage=stage_name,
        epoch=epoch,
        global_step=step,
        loss=float(loss),
        path=path,
    )


def load_checkpoint(model, optimizer, scheduler, path, device,
                    extra_modules: Optional[dict] = None):
    """
    Load checkpoint để resume training.
    Returns: (epoch, step) nếu tìm thấy checkpoint, (0, 0) nếu không.
    """
    if not os.path.exists(path):
        logger.info("Không tìm thấy checkpoint. Bắt đầu training từ đầu.")
        return 0, 0

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if extra_modules:
        extra_state = checkpoint.get('extra_module_state_dicts', {})
        for name, module in extra_modules.items():
            if name in extra_state:
                module.load_state_dict(extra_state[name])
            else:
                logger.warning(f"Checkpoint không có state cho module phụ: {name}")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.current_step = checkpoint['scheduler_step']

    epoch = checkpoint['epoch']
    step = checkpoint['step']
    loss = checkpoint['loss']
    logger.info(f" Checkpoint loaded: {path} (epoch={epoch}, step={step}, loss={loss:.4f})")

    return epoch, step


def load_previous_stage_if_needed(model, current_checkpoint_path: str,
                                  previous_model_path: str, device,
                                  current_stage: str, previous_stage: str):
    """
    Khi chạy từng stage riêng trên cloud, stage hiện tại nên bắt đầu từ stage trước
    nếu chưa có checkpoint riêng của stage hiện tại.
    """
    if os.path.exists(current_checkpoint_path) or not os.path.exists(previous_model_path):
        return

    model.load_state_dict(torch.load(previous_model_path, map_location=device, weights_only=True))
    logger.info(f"{current_stage}: loaded weights from {previous_stage}: {previous_model_path}")


def get_gradient_accumulation_steps() -> int:
    """Số mini-batches tích lũy trước mỗi optimizer step."""
    return max(1, int(TRAIN_CONFIG.get("gradient_accumulation_steps", 1)))


def get_update_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    """Số optimizer updates mỗi epoch sau khi gradient accumulation."""
    return math.ceil(num_batches / accumulation_steps)


def get_accumulation_divisor(batch_idx: int, num_batches: int,
                             accumulation_steps: int) -> int:
    """
    Chia loss theo số batch thực sự trong accumulation block.
    Block cuối có thể ngắn hơn accumulation_steps.
    """
    block_start = (batch_idx // accumulation_steps) * accumulation_steps
    block_end = min(block_start + accumulation_steps, num_batches)
    return block_end - block_start


def should_step_optimizer(batch_idx: int, num_batches: int,
                          accumulation_steps: int) -> bool:
    """True khi đã đủ accumulation block hoặc gặp batch cuối epoch."""
    return ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == num_batches)


def format_duration(seconds: float) -> str:
    """Format ngắn gọn cho ETA/logging."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def get_disk_usage(path: Optional[str]) -> dict:
    """Return disk usage for the filesystem containing path."""
    target = path or "."
    if not os.path.exists(target):
        target = os.path.dirname(target) or "."
    try:
        usage = shutil.disk_usage(target)
        return {
            "disk_total_gb": usage.total / 1e9,
            "disk_used_gb": usage.used / 1e9,
            "disk_free_gb": usage.free / 1e9,
            "disk_used_pct": usage.used / max(usage.total, 1) * 100.0,
        }
    except OSError:
        return {}


def get_progress_log_every_steps() -> int:
    """Tần suất log progress theo batch steps."""
    return max(1, int(TRAIN_CONFIG.get("progress_log_every_steps", 500)))


def get_stage0_time_budget_seconds(debug: bool):
    """Wall-clock budget cho Stage 0; tắt trong debug."""
    if debug:
        return None
    hours = float(TRAIN_CONFIG.get("stage0_time_budget_hours", 0))
    if hours <= 0:
        return None
    return hours * 3600.0


def get_stage0_scheduler_steps(num_batches: int, accumulation_steps: int,
                               time_budget_seconds) -> int:
    """
    Scheduler cần total_steps hữu hạn dù Stage 0 dừng theo timer.
    Dùng expected seconds/batch bảo thủ; nếu chạy nhanh hơn, cosine được clamp ở 0.
    """
    full_update_steps = get_update_steps_per_epoch(num_batches, accumulation_steps)
    if not time_budget_seconds:
        return full_update_steps

    expected_seconds = max(
        1e-6,
        float(TRAIN_CONFIG.get("stage0_scheduler_expected_seconds_per_batch", 0.12))
    )
    expected_batch_steps = max(1, int(time_budget_seconds / expected_seconds))
    expected_update_steps = get_update_steps_per_epoch(expected_batch_steps, accumulation_steps)
    return max(1, min(full_update_steps, expected_update_steps))


def load_stage0_teacher(device):
    """
    Load SentenceTransformer teacher cho Stage 0 distillation.

    Teacher không được train; nó chỉ sinh sentence embeddings làm mục tiêu mềm
    cho student, theo Reimers & Gurevych, EMNLP 2020.
    """
    teacher_name = str(TRAIN_CONFIG.get("stage0_teacher_model"))
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Stage 0 distillation cần package sentence-transformers. "
            "Cài bằng: pip install sentence-transformers."
        ) from exc

    logger.info(f"[Stage0-KD] Loading teacher: {teacher_name}")
    teacher = SentenceTransformer(teacher_name, device=str(device))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    try:
        teacher_dim = teacher.get_sentence_embedding_dimension()
    except Exception:
        teacher_dim = None
    expected_dim = int(MODEL_CONFIG["hidden_size"])
    if teacher_dim is not None and teacher_dim != expected_dim:
        raise ValueError(
            f"Stage 0 direct distillation yêu cầu teacher_dim == student_dim "
            f"({teacher_dim} != {expected_dim}). "
            "Hãy dùng teacher 768d, ví dụ sentence-transformers/all-mpnet-base-v2."
        )
    logger.info(f"[Stage0-KD] Teacher ready | dim={teacher_dim} | device={device}")
    return teacher


def encode_stage0_teacher_embeddings(teacher, tokenizer, input_ids, device):
    """
    Sinh teacher embeddings cho batch hiện tại.

    Dùng batch_decode từ token ids đã có để không phải giữ raw text trong cache
    và không làm phình network volume.
    """
    texts = tokenizer.batch_decode(
        input_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    teacher_batch_size = max(
        1,
        int(TRAIN_CONFIG.get("stage0_teacher_batch_size", len(texts)))
    )
    with torch.no_grad():
        teacher_embeddings = teacher.encode(
            texts,
            batch_size=teacher_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return teacher_embeddings.to(device=device, dtype=torch.float32)


def log_progress(stage_name: str, epoch: int, epochs: int, global_step: int,
                 total_batch_steps: int, epoch_loss: float, num_batches: int,
                 stage_start_time: float, lr: float,
                 time_budget_seconds: Optional[float] = None):
    """Log throughput và ETA để kiểm soát budget cloud."""
    elapsed = max(time.time() - stage_start_time, 1e-9)
    batches_per_sec = num_batches / elapsed
    remaining_batches = max(0, total_batch_steps - global_step)
    dataset_eta = remaining_batches / max(batches_per_sec, 1e-9)
    budget_left = None
    eta = dataset_eta
    if time_budget_seconds is not None and time_budget_seconds > 0:
        budget_left = max(0.0, time_budget_seconds - elapsed)
        eta = min(dataset_eta, budget_left)
    avg_loss = epoch_loss / max(num_batches, 1)
    msg = (
        f"[{stage_name}] Epoch {epoch}/{epochs} | Step {global_step}/{total_batch_steps} | "
        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
        f"{batches_per_sec:.2f} batch/s | Elapsed: {format_duration(elapsed)} | "
        f"ETA: {format_duration(eta)}"
    )
    if budget_left is not None:
        msg += f" | Budget left: {format_duration(budget_left)}"
    logger.info(msg)
    log_event(
        "progress",
        stage=stage_name,
        epoch=epoch,
        epochs=epochs,
        global_step=global_step,
        total_batch_steps=total_batch_steps,
        avg_loss=avg_loss,
        lr=lr,
        batches_per_sec=batches_per_sec,
        elapsed_sec=elapsed,
        eta_sec=eta,
        budget_left_sec=budget_left,
    )


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
                             prefetch_factor=2 if num_workers > 0 else None,
                             drop_last=False)


# =============================================================================
# GIAI ĐOẠN 0: SUPERVISED TEACHER DISTILLATION
# =============================================================================
# Reimers & Gurevych, EMNLP 2020:
#   Student sentence encoder học mimic embedding space của teacher.
#
# Trong pipeline này:
#   1. Teacher all-mpnet-base-v2 sinh normalized 768d sentence embedding.
#   2. Student SWFT sinh 768d sentence embedding từ cùng input.
#   3. Loss = 1 - cosine(normalize(student), normalize(teacher)).
#   4. Teacher chạy eval/no_grad; backward chỉ cập nhật student.

def train_stage0_distillation(model, device, debug=True):
    """
    Giai đoạn 0: supervised direct teacher-student distillation.
    Curriculum Learning Stage 0 (dễ nhất) — Bengio et al., ICML 2009.

    Mục tiêu: Học biểu diễn ngôn ngữ cơ bản từ text thô Wikipedia
    Supervisor: sentence-transformers/all-mpnet-base-v2
    Loss: 1 - cosine(normalize(student), normalize(teacher))
    """
    logger.info("=" * 60)
    logger.info("GIAI ĐOẠN 0: SUPERVISED TEACHER DISTILLATION")
    logger.info("Paper: Reimers & Gurevych, EMNLP 2020")
    logger.info("Teacher target: normalized 768d sentence embedding; backward chỉ qua student")
    logger.info("=" * 60)

    cache_dir = TRAIN_CONFIG.get("data_cache_dir")
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    batch_size = TRAIN_CONFIG["batch_size_debug"] if debug else TRAIN_CONFIG["batch_size_train"]
    num_workers = 0 if device.type != "cuda" else 8  # Wikipedia lớn → cần nhiều workers

    # Dataset
    stage0_max_samples = TRAIN_CONFIG.get("stage0_max_samples") or None
    wiki_dataset = WikipediaDistillationDataset(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        max_length=MODEL_CONFIG["max_seq_length"],
        debug=debug,
        num_debug_samples=DEBUG_CONFIG["num_samples"],
        max_samples=stage0_max_samples,
        sample_offset=TRAIN_CONFIG.get("stage0_sample_offset", 0)
    )
    train_loader = create_dataloader(wiki_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     prefetch_factor=2 if num_workers > 0 else None)
    accumulation_steps = get_gradient_accumulation_steps()

    # Evaluation
    eval_loader = _create_eval_loader(device, debug, cache_dir)

    stage_log_name = "Stage0-distillation"

    # Loss
    teacher_model = load_stage0_teacher(device)
    distillation_criterion = Stage0TeacherDistillationLoss()
    logger.info(
        "[Stage0-KD] Enabled | objective=distillation | teacher=%s | weight=%.3f",
        TRAIN_CONFIG["stage0_teacher_model"],
        TRAIN_CONFIG["stage0_distillation_weight"],
    )

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
    total_batch_steps = len(train_loader) * epochs
    time_budget_seconds = get_stage0_time_budget_seconds(debug)
    total_steps = get_stage0_scheduler_steps(
        total_batch_steps, accumulation_steps, time_budget_seconds
    )
    warmup_steps = int(TRAIN_CONFIG["warmup_ratio"] * total_steps)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # Mixed Precision
    use_amp = TRAIN_CONFIG["use_amp_on_cuda"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Resume
    checkpoint_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage0_latest.pt")
    start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)

    log_every = get_progress_log_every_steps()
    checkpoint_interval_seconds = max(
        60.0,
        float(TRAIN_CONFIG.get("checkpoint_every_minutes", 30)) * 60.0
    )

    logger.info(f"Wikipedia sentences: {len(wiki_dataset):,}")
    logger.info(f"Epochs: {epochs}, Batch steps: {total_batch_steps}, "
                f"Scheduler optimizer steps: {total_steps}, Warmup: {warmup_steps}")
    logger.info(f"Batch size: {batch_size}, Grad accumulation: {accumulation_steps}, "
                f"Optimizer effective batch: {batch_size * accumulation_steps}, "
                f"AMP: {use_amp}, Device: {device}")
    logger.info(f"Direct KD teacher: {TRAIN_CONFIG['stage0_teacher_model']}")
    logger.info("Stage0 objective: supervised distillation")
    if not debug:
        sample_cap_text = f"{stage0_max_samples:,}" if stage0_max_samples else "none"
        budget_text = format_duration(time_budget_seconds) if time_budget_seconds else "disabled"
        logger.info(f"Budget target: {TRAIN_CONFIG.get('target_train_hours')}h total train | "
                    f"Stage0 time budget: {budget_text} | Stage0 sample cap: {sample_cap_text}")
    log_event(
        "stage_start",
        stage=stage_log_name,
        epochs=epochs,
        total_batch_steps=total_batch_steps,
        scheduler_steps=total_steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        use_amp=use_amp,
        device=str(device),
        debug=debug,
        time_budget_sec=time_budget_seconds,
        teacher_model=TRAIN_CONFIG["stage0_teacher_model"],
        **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
    )

    # ===== TRAINING LOOP =====
    model.train()
    global_step = start_step
    stage_start_time = time.time()
    last_checkpoint_time = stage_start_time
    stop_stage0 = False

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            elapsed = time.time() - stage_start_time
            if time_budget_seconds is not None and elapsed >= time_budget_seconds:
                logger.info(f"[{stage_log_name}] Time budget reached after "
                            f"{format_duration(elapsed)} at step {global_step}.")
                stop_stage0 = True
                break

            if epoch == start_epoch and batch_idx < (start_step % len(train_loader)):
                continue

            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            teacher_embeddings = encode_stage0_teacher_embeddings(
                teacher_model, tokenizer, ids, device
            )

            with autocast(device_type=device.type, enabled=use_amp):
                model.train()  # Đảm bảo Dropout BẬT
                embedding_1 = model(ids, mask)
                distillation_loss = distillation_criterion(
                    embedding_1,
                    teacher_embeddings
                )
                loss = TRAIN_CONFIG["stage0_distillation_weight"] * distillation_loss

            raw_loss = loss.detach()
            loss = loss / get_accumulation_divisor(batch_idx, len(train_loader), accumulation_steps)
            scaler.scale(loss).backward()

            if should_step_optimizer(batch_idx, len(train_loader), accumulation_steps):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += raw_loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                log_progress(
                    stage_log_name, epoch + 1, epochs, global_step,
                    total_batch_steps, epoch_loss, num_batches,
                    stage_start_time, scheduler.get_lr(),
                    time_budget_seconds=time_budget_seconds
                )

            now = time.time()
            if now - last_checkpoint_time >= checkpoint_interval_seconds:
                avg_loss = epoch_loss / max(num_batches, 1)
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, avg_loss,
                    checkpoint_path,
                    stage_name=stage_log_name
                )
                last_checkpoint_time = now

        # Evaluate
        spearman = evaluate_stsb(model, eval_loader, device)
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"[{stage_log_name}] Epoch {epoch+1}/{epochs} DONE | "
                   f"Avg Loss: {avg_loss:.4f} | STS-B Spearman: {spearman:.4f}")
        log_event(
            "epoch_end",
            stage=stage_log_name,
            epoch=epoch + 1,
            epochs=epochs,
            global_step=global_step,
            avg_loss=avg_loss,
            stsb_spearman=spearman,
            lr=scheduler.get_lr(),
            elapsed_sec=time.time() - stage_start_time,
            **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
        )

        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_loss,
                       checkpoint_path, stage_name=stage_log_name)

        if stop_stage0:
            break

    final_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage0_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f" Giai đoạn 0 hoàn thành. Model saved: {final_path}")
    log_event(
        "stage_end",
        stage=stage_log_name,
        final_model_path=final_path,
        global_step=global_step,
        elapsed_sec=time.time() - stage_start_time,
        **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
    )


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
    accumulation_steps = get_gradient_accumulation_steps()

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
    update_steps_per_epoch = get_update_steps_per_epoch(len(train_loader), accumulation_steps)
    total_steps = update_steps_per_epoch * epochs
    warmup_steps = int(TRAIN_CONFIG["warmup_ratio"] * total_steps)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # Mixed Precision — FP16 trên CUDA, FP32 trên MPS
    use_amp = TRAIN_CONFIG["use_amp_on_cuda"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Resume from checkpoint
    checkpoint_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage1_latest.pt")
    previous_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage0_final.pt")
    load_previous_stage_if_needed(model, checkpoint_path, previous_path, device,
                                  "Stage1", "Stage0")
    start_epoch, start_step = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path, device,
        extra_modules={'criterion': criterion}
    )

    total_batch_steps = len(train_loader) * epochs
    log_every = get_progress_log_every_steps()

    logger.info(f"Epochs: {epochs}, Batch steps: {total_batch_steps}, "
                f"Optimizer steps: {total_steps}, Warmup: {warmup_steps}")
    logger.info(f"Batch size: {batch_size}, Grad accumulation: {accumulation_steps}, "
                f"Optimizer effective batch: {batch_size * accumulation_steps}, "
                f"AMP: {use_amp}, Device: {device}")
    log_event(
        "stage_start",
        stage="Stage1",
        epochs=epochs,
        total_batch_steps=total_batch_steps,
        scheduler_steps=total_steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        use_amp=use_amp,
        device=str(device),
        debug=debug,
        **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
    )

    # ===== TRAINING LOOP =====
    model.train()
    global_step = start_step
    stage_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

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
            with autocast(device_type=device.type, enabled=use_amp):
                # Encode 2 câu qua Siamese Encoder (chia sẻ trọng số)
                embedding_a = model(ids_a, mask_a)  # (B, H)
                embedding_b = model(ids_b, mask_b)  # (B, H)

                # SoftmaxLoss: concat(u, v, |u-v|) → classifier → CE
                loss = criterion(embedding_a, embedding_b, labels)

            # ===== Backward Pass =====
            raw_loss = loss.detach()
            loss = loss / get_accumulation_divisor(batch_idx, len(train_loader), accumulation_steps)
            scaler.scale(loss).backward()

            if should_step_optimizer(batch_idx, len(train_loader), accumulation_steps):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += raw_loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                log_progress(
                    "Stage1", epoch + 1, epochs, global_step,
                    total_batch_steps, epoch_loss, num_batches,
                    stage_start_time, scheduler.get_lr()
                )

        # Evaluate after each epoch
        spearman = evaluate_stsb(model, eval_loader, device)
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"[Stage1] Epoch {epoch+1}/{epochs} DONE | "
                   f"Avg Loss: {avg_loss:.4f} | STS-B Spearman: {spearman:.4f}")
        log_event(
            "epoch_end",
            stage="Stage1",
            epoch=epoch + 1,
            epochs=epochs,
            global_step=global_step,
            avg_loss=avg_loss,
            stsb_spearman=spearman,
            lr=scheduler.get_lr(),
            elapsed_sec=time.time() - stage_start_time,
            **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
        )

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_loss,
                       checkpoint_path, extra_modules={'criterion': criterion},
                       stage_name="Stage1")

    # Save final model
    final_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage1_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f" Giai đoạn 1 hoàn thành. Model saved: {final_path}")
    log_event(
        "stage_end",
        stage="Stage1",
        final_model_path=final_path,
        global_step=global_step,
        elapsed_sec=time.time() - stage_start_time,
        **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
    )


# =============================================================================
# GIAI ĐOẠN 2: SIMILARITY TRAINING (Contrastive Learning)
# =============================================================================

def train_stage2_similarity(model, device, debug=True, include_hard_negatives=False):
    """
    Giai đoạn 2: Similarity Fine-tune — Reimers & Gurevych, EMNLP 2019.
    Curriculum Learning Stage 2 (khó hơn) — Bengio et al., ICML 2009.

    Mục tiêu: Tối ưu embedding cho similarity/paraphrase.
    - Stage2: SimCSE-style contrastive softmax trên positive pairs.
    - Stage2+HN: SBERT SoftmaxLoss 2 lớp trên cặp PAWS paraphrase/non-paraphrase.
    """
    stage_name = "Stage2+HN" if include_hard_negatives else "Stage2"
    logger.info("=" * 60)
    logger.info(f"GIAI ĐOẠN 2: SIMILARITY TRAINING ({stage_name})")
    logger.info("Paper: Gao et al., EMNLP 2021 (contrastive) + Zhang et al., NAACL 2019 (PAWS)")
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
    accumulation_steps = get_gradient_accumulation_steps()

    # Evaluation
    eval_loader = _create_eval_loader(device, debug, cache_dir)

    # Loss
    if include_hard_negatives:
        # PAWS cung cấp pair labels, không phải triplets cùng anchor.
        # Dùng supervised SBERT-style pair classification trên [u, v, |u-v|].
        criterion = SoftmaxLoss(hidden_size=MODEL_CONFIG["hidden_size"], num_labels=2).to(device)
    else:
        criterion = MultipleNegativesRankingLoss(temperature=TRAIN_CONFIG["temperature"])

    # Optimizer — AdamW
    optim_params = list(model.parameters())
    if include_hard_negatives:
        optim_params += list(criterion.parameters())
    optimizer = AdamW(
        optim_params,
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(TRAIN_CONFIG["adam_beta1"], TRAIN_CONFIG["adam_beta2"]),
        eps=TRAIN_CONFIG["adam_epsilon"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )

    # LR Schedule
    epochs = TRAIN_CONFIG["epochs_stage2"]
    update_steps_per_epoch = get_update_steps_per_epoch(len(train_loader), accumulation_steps)
    total_steps = update_steps_per_epoch * epochs
    warmup_steps = int(TRAIN_CONFIG["warmup_ratio"] * total_steps)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # Mixed Precision
    use_amp = TRAIN_CONFIG["use_amp_on_cuda"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Resume from checkpoint
    ckpt_name = "stage2_hn_latest.pt" if include_hard_negatives else "stage2_latest.pt"
    checkpoint_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], ckpt_name)
    previous_name = "stage2_final.pt" if include_hard_negatives else "stage1_final.pt"
    previous_stage = "Stage2" if include_hard_negatives else "Stage1"
    previous_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], previous_name)
    load_previous_stage_if_needed(model, checkpoint_path, previous_path, device,
                                  stage_name, previous_stage)
    extra_modules = {'criterion': criterion} if include_hard_negatives else None
    start_epoch, start_step = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path, device,
        extra_modules=extra_modules
    )

    total_batch_steps = len(train_loader) * epochs
    log_every = get_progress_log_every_steps()

    logger.info(f"Epochs: {epochs}, Batch steps: {total_batch_steps}, "
                f"Optimizer steps: {total_steps}, Warmup: {warmup_steps}")
    if include_hard_negatives:
        logger.info(f"Batch size: {batch_size}, Grad accumulation: {accumulation_steps}, "
                    f"Optimizer effective batch: {batch_size * accumulation_steps}, "
                    "Objective: PAWS binary SoftmaxLoss")
    else:
        logger.info(f"Batch size: {batch_size}, Grad accumulation: {accumulation_steps}, "
                    f"Optimizer effective batch: {batch_size * accumulation_steps}, "
                    f"Contrastive negatives per batch: {batch_size - 1}, "
                    f"Temperature: {TRAIN_CONFIG['temperature']}")
    log_event(
        "stage_start",
        stage=stage_name,
        epochs=epochs,
        total_batch_steps=total_batch_steps,
        scheduler_steps=total_steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        use_amp=use_amp,
        device=str(device),
        debug=debug,
        hard_negatives=include_hard_negatives,
        **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
    )

    # ===== TRAINING LOOP =====
    model.train()
    global_step = start_step
    stage_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < (start_step % len(train_loader)):
                continue

            ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            ids_b = batch['input_ids_b'].to(device)
            mask_b = batch['attention_mask_b'].to(device)
            labels = batch['label'].to(device) if include_hard_negatives else None

            with autocast(device_type=device.type, enabled=use_amp):
                # Encode anchor và positive
                embedding_a = model(ids_a, mask_a)  # (B, H)
                embedding_b = model(ids_b, mask_b)  # (B, H)

                if include_hard_negatives:
                    # Supervised PAWS pair classification: paraphrase vs non-paraphrase.
                    loss = criterion(embedding_a, embedding_b, labels)
                else:
                    # Contrastive loss: cosine similarity matrix + cross-entropy
                    loss = criterion(embedding_a, embedding_b)

            raw_loss = loss.detach()
            loss = loss / get_accumulation_divisor(batch_idx, len(train_loader), accumulation_steps)
            scaler.scale(loss).backward()

            if should_step_optimizer(batch_idx, len(train_loader), accumulation_steps):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += raw_loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                log_progress(
                    stage_name, epoch + 1, epochs, global_step,
                    total_batch_steps, epoch_loss, num_batches,
                    stage_start_time, scheduler.get_lr()
                )

        spearman = evaluate_stsb(model, eval_loader, device)
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"[{stage_name}] Epoch {epoch+1}/{epochs} DONE | "
                   f"Avg Loss: {avg_loss:.4f} | STS-B Spearman: {spearman:.4f}")
        log_event(
            "epoch_end",
            stage=stage_name,
            epoch=epoch + 1,
            epochs=epochs,
            global_step=global_step,
            avg_loss=avg_loss,
            stsb_spearman=spearman,
            lr=scheduler.get_lr(),
            elapsed_sec=time.time() - stage_start_time,
            hard_negatives=include_hard_negatives,
            **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
        )

        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_loss,
                       checkpoint_path, extra_modules=extra_modules,
                       stage_name=stage_name)

    final_name = "stage2_hn_final.pt" if include_hard_negatives else "stage2_final.pt"
    final_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], final_name)
    torch.save(model.state_dict(), final_path)
    logger.info(f" Giai đoạn 2 ({stage_name}) hoàn thành. Model saved: {final_path}")
    log_event(
        "stage_end",
        stage=stage_name,
        final_model_path=final_path,
        global_step=global_step,
        elapsed_sec=time.time() - stage_start_time,
        hard_negatives=include_hard_negatives,
        **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
    )


# =============================================================================
# MAIN — Pipeline hoàn chỉnh
# =============================================================================

def main():
    device = get_device()
    debug = DEBUG_CONFIG["enabled"]
    metrics_log_path = str(TRAIN_CONFIG["metrics_log_path"])
    init_training_logger(metrics_log_path)

    logger.info("=" * 60)
    logger.info("SWFT — Shallow-Wide Factorized Transformer")
    logger.info("Train from scratch — Pure PyTorch")
    logger.info(f"Device: {device} | Debug: {debug}")
    logger.info(f"Metrics JSONL: {metrics_log_path}")
    logger.info("=" * 60)

    # Tạo model
    model = create_swft_model(MODEL_CONFIG).to(device)
    total_params = model.count_parameters()
    logger.info(f"Tổng tham số: {total_params:,} (~{total_params/1e6:.1f}M)")
    log_event(
        "training_start",
        device=str(device),
        debug=debug,
        total_params=total_params,
        metrics_log_path=metrics_log_path,
        checkpoint_dir=TRAIN_CONFIG["checkpoint_dir"],
        data_cache_dir=TRAIN_CONFIG["data_cache_dir"],
        target_train_hours=TRAIN_CONFIG["target_train_hours"],
        stage0_time_budget_hours=TRAIN_CONFIG["stage0_time_budget_hours"],
        batch_size=TRAIN_CONFIG["batch_size_debug"] if debug else TRAIN_CONFIG["batch_size_train"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        stage0_teacher_model=TRAIN_CONFIG["stage0_teacher_model"],
        **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
    )

    # ===== Curriculum Learning (Bengio et al., ICML 2009) =====
    # Stage 0: supervised teacher-student distillation từ text thô
    # Stage 1: NLI (dễ — phân loại 3 classes)
    # Stage 2: Similarity + Hard Negatives (khó nhất — contrastive)

    # Giai đoạn 0: supervised distillation trên Wikipedia
    train_stage0_distillation(model, device, debug=debug)

    # Giai đoạn 1: NLI
    train_stage1_nli(model, device, debug=debug)

    # Giai đoạn 2: Similarity contrastive learning trên PAWS
    train_stage2_similarity(model, device, debug=debug, include_hard_negatives=False)

    # Giai đoạn 2+: Similarity + Hard Negatives (PAWS adversarial)
    train_stage2_similarity(model, device, debug=debug, include_hard_negatives=True)

    logger.info("🎉 HOÀN THÀNH TOÀN BỘ PIPELINE TRAINING!")
    log_event(
        "training_complete",
        checkpoint_dir=TRAIN_CONFIG["checkpoint_dir"],
        **get_disk_usage(TRAIN_CONFIG["checkpoint_dir"]),
    )


if __name__ == "__main__":
    main()
