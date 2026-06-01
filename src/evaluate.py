# =============================================================================
# evaluate.py — Đánh giá mô hình SWFT trên STS-B, PAWS
# =============================================================================
# Bài báo tham khảo:
#   [1] Cer et al., "STS Benchmark", SemEval@ACL 2017 → Spearman correlation
#   [2] Zhang et al., "PAWS", NAACL 2019 → Adversarial accuracy
#   [3] Muennighoff et al., "MTEB", EACL 2023 → Evaluation methodology
# =============================================================================

import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, TRAIN_CONFIG, DEBUG_CONFIG, DATA_CONFIG, get_device
from model.sbert import create_swft_model
from dataset import get_tokenizer, STSBDataset, PAWSEvalDataset, create_dataloader

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_stsb_full(model, device, debug=True):
    """
    Đánh giá trên STS Benchmark — Cer et al., SemEval@ACL 2017.
    Metric chính: Spearman Correlation ρ (rank correlation)
    """
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    batch_size = TRAIN_CONFIG["batch_size_debug"] if debug else TRAIN_CONFIG["batch_size_train"]

    dataset = STSBDataset(tokenizer, split="test", debug=debug,
                          num_debug_samples=DEBUG_CONFIG["eval_samples"])
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_cosine = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            ids_b = batch['input_ids_b'].to(device)
            mask_b = batch['attention_mask_b'].to(device)

            emb_a = model(ids_a, mask_a)
            emb_b = model(ids_b, mask_b)

            cos_sim = nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)
            all_cosine.extend(cos_sim.cpu().tolist())
            all_labels.extend(batch['score'].tolist())

    spearman, p_value = spearmanr(all_cosine, all_labels)
    logger.info(f" STS-B Test | Spearman ρ = {spearman:.4f} (p={p_value:.2e})")
    return spearman


def evaluate_paws(model, device, debug=True):
    """
    Đánh giá trên PAWS — Zhang et al., NAACL 2019.
    Metric: Accuracy trên adversarial paraphrases.
    BM25 thường chỉ đạt ~50% trên tập này (vì từ vựng giống nhau).
    """
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    batch_size = TRAIN_CONFIG["batch_size_debug"] if debug else TRAIN_CONFIG["batch_size_train"]

    dataset = PAWSEvalDataset(tokenizer, debug=debug,
                              num_debug_samples=DEBUG_CONFIG["eval_samples"])
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_cosine = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            ids_a = batch['input_ids_a'].to(device)
            mask_a = batch['attention_mask_a'].to(device)
            ids_b = batch['input_ids_b'].to(device)
            mask_b = batch['attention_mask_b'].to(device)

            emb_a = model(ids_a, mask_a)
            emb_b = model(ids_b, mask_b)

            cos_sim = nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)
            all_cosine.extend(cos_sim.cpu().tolist())
            all_labels.extend(batch['label'].tolist())

    # Tìm threshold tối ưu cho binary classification
    best_acc = 0.0
    best_threshold = 0.0
    for threshold in np.arange(0.0, 1.0, 0.01):
        preds = [1 if s >= threshold else 0 for s in all_cosine]
        acc = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    logger.info(f" PAWS Test | Accuracy = {best_acc:.4f} (threshold={best_threshold:.2f})")
    return best_acc


def main():
    device = get_device()
    debug = DEBUG_CONFIG["enabled"]

    logger.info("=" * 60)
    logger.info("ĐÁNH GIÁ MÔ HÌNH SWFT")
    logger.info("=" * 60)

    # Tạo model
    model = create_swft_model(MODEL_CONFIG).to(device)

    # Thử load checkpoint
    model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage2_hn_final.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage2_final.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage1_final.pt")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded model: {model_path}")
    else:
        logger.warning("Không tìm thấy checkpoint. Đánh giá model chưa train (random weights).")

    # Đánh giá
    evaluate_stsb_full(model, device, debug=debug)
    evaluate_paws(model, device, debug=debug)

    logger.info(" Đánh giá hoàn thành!")


if __name__ == "__main__":
    main()
