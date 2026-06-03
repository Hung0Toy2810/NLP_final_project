# =============================================================================
# prepare_data.py — Tải & Pre-tokenize toàn bộ dữ liệu TRƯỚC KHI training
# =============================================================================
# Script này chạy ĐỘC LẬP, MỘT LẦN DUY NHẤT trước khi bắt đầu train.
# Mục đích: loại bỏ hoàn toàn CPU bottleneck khi GPU đang train.
#
# Quy trình:
#   1. Tải raw text từ HuggingFace (Wikipedia, SNLI, PAWS, STS-B)
#   2. Tokenize toàn bộ text → mảng số (input_ids, attention_mask)
#   3. Lưu xuống ổ cứng dưới dạng Apache Arrow (zero-copy memory mapping)
#   4. Khi train, DataLoader chỉ đọc mảng số — GPU không phải chờ CPU
#
# Bài báo tham khảo:
#   [1] Reimers & Gurevych, EMNLP 2020 — teacher-student distillation
#   [2] Bowman et al., EMNLP 2015 — SNLI dataset
#   [3] Zhang et al., NAACL 2019 — PAWS dataset
#   [4] Cer et al., SemEval@ACL 2017 — STS Benchmark
#
# Tối ưu I/O:
#   - Dùng dataset.map() với num_proc (multi-processing) để tokenize song song
#   - Dùng dataset.save_to_disk() lưu Apache Arrow format (zero-copy read)
#   - Trên Vast.ai H100: CPU 32-64 cores → tokenize xong < 10 phút cho 20GB
# =============================================================================

import os
import sys
import time
import logging
import argparse
from typing import cast

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset, IterableDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Đường dẫn lưu cache (trên ổ cứng local NVMe của Vast.ai)
DEFAULT_CACHE_DIR = "./data_cache"
MAX_SEQ_LENGTH = 128


def get_num_proc():
    """Xác định số CPU cores để chạy song song."""
    try:
        n = os.cpu_count()
        # Giữ lại 2 cores cho hệ thống
        return max(1, n - 2) if n else 1
    except Exception:
        return 1


# =============================================================================
# 1. WIKIPEDIA — cho Stage 0 (Teacher-Student Distillation)
# =============================================================================
# Reimers & Gurevych, EMNLP 2020:
#   student sentence encoder học mimic embedding space của teacher.
# =============================================================================

def prepare_wikipedia(tokenizer, cache_dir: str, debug: bool = False):
    """
    Tải Wikipedia English, tách thành từng câu, tokenize.

    Wikipedia gốc là các bài viết dài (article-level). Chúng ta cần tách
    thành từng câu (sentence-level) vì SWFT xử lý sentence embedding.

    Output lưu tại: {cache_dir}/wikipedia_tokenized/
    """
    save_path = os.path.join(cache_dir, "wikipedia_tokenized")
    if os.path.exists(save_path):
        logger.info(f" Wikipedia cache đã tồn tại: {save_path}")
        return

    logger.info("=" * 60)
    logger.info("CHUẨN BỊ WIKIPEDIA (Stage 0 — Teacher Distillation, EMNLP 2020)")
    logger.info("=" * 60)

    start = time.time()

    # Tải Wikipedia — wikimedia/wikipedia là phiên bản mới nhất trên HuggingFace
    logger.info("Đang tải Wikipedia English từ HuggingFace...")
    if debug:
        # Debug: sử dụng streaming=True để tránh tải toàn bộ 20GB về máy
        logger.info("[DEBUG] Đang tải Wikipedia ở chế độ Streaming (chỉ lấy 10,000 articles)...")
        wiki_stream = cast(IterableDataset, load_dataset(
            "wikimedia/wikipedia", "20231101.en",
            split="train",
            streaming=True
        ))
        # Chỉ lấy 10,000 articles đầu tiên
        wiki_stream = wiki_stream.take(10000)
        
        # Chuyển đổi thành Dataset thông thường để hỗ trợ len(), num_proc và save_to_disk
        logger.info("[DEBUG] Đang chuyển đổi stream thành Dataset thông thường...")
        wiki = cast(Dataset, Dataset.from_generator(
            lambda: (yield from wiki_stream),
            features=wiki_stream.features
        ))
        logger.info(f"[DEBUG] Tải {len(wiki)} articles")
    else:
        wiki = cast(Dataset, load_dataset("wikimedia/wikipedia", "20231101.en", split="train"))
        logger.info(f"Tải {len(wiki)} articles (đầy đủ)")

    # Type assertion: load_dataset với split cụ thể luôn trả về Dataset (map-style),
    # KHÔNG phải IterableDataset. Assert này giúp linter hiểu đúng kiểu dữ liệu.
    assert isinstance(wiki, Dataset), f"Expected Dataset, got {type(wiki)}"

    # ===== Bước 1: Tách article → sentences =====
    # Wikipedia articles dài hàng ngàn từ. Mô hình SWFT cần sentence-level.
    # Dùng phương pháp đơn giản: split bằng dấu chấm câu.
    logger.info("Đang tách articles thành sentences...")

    def extract_sentences(batch):
        """Tách mỗi article thành các câu riêng lẻ."""
        all_sentences = []
        for text in batch['text']:
            # Tách bằng dấu chấm câu, lọc câu quá ngắn (< 20 ký tự)
            sentences = text.split('. ')
            for sent in sentences:
                sent = sent.strip()
                # Chỉ giữ câu có độ dài phù hợp (20-500 ký tự)
                # Quá ngắn: vô nghĩa (ví dụ: "See also")
                # Quá dài: sẽ bị truncate, lãng phí tokens
                if 20 <= len(sent) <= 500:
                    all_sentences.append(sent)
        return {'sentence': all_sentences}

    sentences_dataset = wiki.map(
        extract_sentences,
        batched=True,
        batch_size=1000,
        remove_columns=wiki.column_names,
        num_proc=get_num_proc(),
        desc="Tách sentences"
    )

    logger.info(f"Tổng số câu sau khi tách: {len(sentences_dataset):,}")

    # ===== Bước 2: Tokenize =====
    logger.info("Đang tokenize (multi-process)...")

    def tokenize_sentence(batch):
        """Tokenize một batch câu."""
        encoded = tokenizer(
            batch['sentence'],
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
        )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }

    tokenized = sentences_dataset.map(
        tokenize_sentence,
        batched=True,
        batch_size=5000,
        remove_columns=['sentence'],
        num_proc=get_num_proc(),
        desc="Tokenize Wikipedia"
    )

    # Đặt format PyTorch để DataLoader đọc trực tiếp
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # ===== Bước 3: Lưu xuống ổ cứng (Apache Arrow) =====
    tokenized.save_to_disk(save_path)
    elapsed = time.time() - start
    logger.info(f" Wikipedia tokenized & saved: {save_path}")
    logger.info(f"   Số câu: {len(tokenized):,} | Thời gian: {elapsed:.0f}s")


# =============================================================================
# 2. SNLI — cho Stage 1 (NLI SoftmaxLoss)
# =============================================================================
# Bowman et al., "A large annotated corpus for learning NLI", EMNLP 2015

def prepare_snli(tokenizer, cache_dir: str, debug: bool = False):
    """
    Tải SNLI, lọc unlabeled samples, tokenize cả premise và hypothesis.
    Output: {cache_dir}/snli_tokenized/
    """
    save_path = os.path.join(cache_dir, "snli_tokenized")
    if os.path.exists(save_path):
        logger.info(f" SNLI cache đã tồn tại: {save_path}")
        return

    logger.info("=" * 60)
    logger.info("CHUẨN BỊ SNLI (Stage 1 — NLI, Bowman et al., EMNLP 2015)")
    logger.info("=" * 60)

    start = time.time()

    # Tải SNLI
    logger.info("Đang tải SNLI từ HuggingFace...")
    snli = load_dataset("stanfordnlp/snli")

    # Type assertion: giúp linter (Pylance) hiểu snli là DatasetDict, không phải IterableDataset
    assert isinstance(snli, DatasetDict), f"Expected DatasetDict, got {type(snli)}"

    # Lọc bỏ samples có label = -1 (unlabeled / broken)
    def filter_valid(example):
        return example['label'] != -1

    snli = snli.filter(filter_valid, num_proc=get_num_proc(), desc="Lọc label=-1")

    if debug:
        snli = DatasetDict({
            'train': snli['train'].select(range(min(5000, len(snli['train'])))),
            'validation': snli['validation'].select(range(min(1000, len(snli['validation'])))),
            'test': snli['test'].select(range(min(1000, len(snli['test'])))),
        })
        logger.info(f"[DEBUG] Train: {len(snli['train'])}, Val: {len(snli['validation'])}")

    # Tokenize
    logger.info("Đang tokenize SNLI (premise + hypothesis riêng biệt)...")

    def tokenize_nli(batch):
        """Tokenize premise và hypothesis riêng biệt (không pair)."""
        enc_premise = tokenizer(
            batch['premise'],
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
        )
        enc_hypothesis = tokenizer(
            batch['hypothesis'],
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
        )
        return {
            'input_ids_a': enc_premise['input_ids'],
            'attention_mask_a': enc_premise['attention_mask'],
            'input_ids_b': enc_hypothesis['input_ids'],
            'attention_mask_b': enc_hypothesis['attention_mask'],
            'label': batch['label'],
        }

    tokenized = snli.map(
        tokenize_nli,
        batched=True,
        batch_size=5000,
        remove_columns=['premise', 'hypothesis'],
        num_proc=get_num_proc(),
        desc="Tokenize SNLI"
    )

    tokenized.set_format(type='torch',
                         columns=['input_ids_a', 'attention_mask_a',
                                  'input_ids_b', 'attention_mask_b', 'label'])
    tokenized.save_to_disk(save_path)

    elapsed = time.time() - start
    logger.info(f" SNLI tokenized & saved: {save_path}")
    logger.info(f"   Train: {len(tokenized['train']):,} | Thời gian: {elapsed:.0f}s")


# =============================================================================
# 3. PAWS — cho Stage 2 (Contrastive Loss + Hard Negatives)
# =============================================================================
# Zhang et al., "PAWS: Paraphrase Adversaries from Word Scrambling", NAACL 2019

def prepare_paws(tokenizer, cache_dir: str, debug: bool = False):
    """
    Tải PAWS, tokenize sentence1 và sentence2.
    Output: {cache_dir}/paws_tokenized/
    """
    save_path = os.path.join(cache_dir, "paws_tokenized")
    if os.path.exists(save_path):
        logger.info(f" PAWS cache đã tồn tại: {save_path}")
        return

    logger.info("=" * 60)
    logger.info("CHUẨN BỊ PAWS (Stage 2 — Hard Negatives, Zhang et al., NAACL 2019)")
    logger.info("=" * 60)

    start = time.time()

    logger.info("Đang tải PAWS từ HuggingFace...")
    paws = load_dataset("google-research-datasets/paws", "labeled_final")
    assert isinstance(paws, DatasetDict), f"Expected DatasetDict, got {type(paws)}"

    if debug:
        paws = DatasetDict({
            'train': paws['train'].select(range(min(5000, len(paws['train'])))),
            'validation': paws['validation'].select(range(min(1000, len(paws['validation'])))),
            'test': paws['test'].select(range(min(1000, len(paws['test'])))),
        })

    logger.info("Đang tokenize PAWS...")

    def tokenize_paws(batch):
        enc1 = tokenizer(batch['sentence1'], max_length=MAX_SEQ_LENGTH,
                         padding='max_length', truncation=True)
        enc2 = tokenizer(batch['sentence2'], max_length=MAX_SEQ_LENGTH,
                         padding='max_length', truncation=True)
        return {
            'input_ids_a': enc1['input_ids'],
            'attention_mask_a': enc1['attention_mask'],
            'input_ids_b': enc2['input_ids'],
            'attention_mask_b': enc2['attention_mask'],
            'label': batch['label'],
        }

    tokenized = paws.map(
        tokenize_paws,
        batched=True,
        batch_size=5000,
        remove_columns=['sentence1', 'sentence2', 'id'],
        num_proc=get_num_proc(),
        desc="Tokenize PAWS"
    )

    tokenized.set_format(type='torch',
                         columns=['input_ids_a', 'attention_mask_a',
                                  'input_ids_b', 'attention_mask_b', 'label'])
    tokenized.save_to_disk(save_path)

    elapsed = time.time() - start
    logger.info(f" PAWS tokenized & saved: {save_path}")
    logger.info(f"   Train: {len(tokenized['train']):,} | Thời gian: {elapsed:.0f}s")


# =============================================================================
# 4. STS-B — cho Evaluation
# =============================================================================
# Cer et al., "STS Benchmark", SemEval@ACL 2017

def prepare_stsb(tokenizer, cache_dir: str, debug: bool = False):
    """
    Tải STS-B, tokenize, normalize score 0-5 → 0-1.
    Output: {cache_dir}/stsb_tokenized/
    """
    save_path = os.path.join(cache_dir, "stsb_tokenized")
    if os.path.exists(save_path):
        logger.info(f" STS-B cache đã tồn tại: {save_path}")
        return

    logger.info("=" * 60)
    logger.info("CHUẨN BỊ STS-B (Evaluation — Cer et al., SemEval@ACL 2017)")
    logger.info("=" * 60)

    start = time.time()

    logger.info("Đang tải STS-B từ HuggingFace...")
    stsb = load_dataset("mteb/stsbenchmark-sts")
    assert isinstance(stsb, DatasetDict), f"Expected DatasetDict, got {type(stsb)}"

    if debug:
        stsb = DatasetDict({
            split: stsb[split].select(range(min(500, len(stsb[split]))))
            for split in stsb.keys()
        })

    def tokenize_stsb(batch):
        enc1 = tokenizer(batch['sentence1'], max_length=MAX_SEQ_LENGTH,
                         padding='max_length', truncation=True)
        enc2 = tokenizer(batch['sentence2'], max_length=MAX_SEQ_LENGTH,
                         padding='max_length', truncation=True)
        # Normalize score 0-5 → 0-1
        scores = [s / 5.0 for s in batch['score']]
        return {
            'input_ids_a': enc1['input_ids'],
            'attention_mask_a': enc1['attention_mask'],
            'input_ids_b': enc2['input_ids'],
            'attention_mask_b': enc2['attention_mask'],
            'score': scores,
        }

    tokenized = stsb.map(
        tokenize_stsb,
        batched=True,
        batch_size=5000,
        remove_columns=['sentence1', 'sentence2'],
        num_proc=get_num_proc(),
        desc="Tokenize STS-B"
    )

    tokenized.save_to_disk(save_path)

    elapsed = time.time() - start
    logger.info(f" STS-B tokenized & saved: {save_path}")
    logger.info(f"   Thời gian: {elapsed:.0f}s")


# =============================================================================
# MAIN — Chạy toàn bộ pipeline chuẩn bị dữ liệu
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tải & Pre-tokenize dữ liệu cho SWFT")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: chỉ tải một phần nhỏ dữ liệu")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Thư mục lưu dữ liệu đã tokenize")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    logger.info(" BẮT ĐẦU CHUẨN BỊ DỮ LIỆU")
    logger.info(f"   Cache dir: {args.cache_dir}")
    logger.info(f"   Debug mode: {args.debug}")
    logger.info(f"   CPU cores: {get_num_proc()}")

    # Load tokenizer một lần duy nhất
    logger.info("Đang load AutoTokenizer fast (bert-base-uncased)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    logger.info(f"   Vocab size: {tokenizer.vocab_size}")

    total_start = time.time()

    # ===== Chuẩn bị từng dataset =====
    # Thứ tự: Wikipedia (lớn nhất, lâu nhất) → SNLI → PAWS → STS-B
    prepare_wikipedia(tokenizer, args.cache_dir, debug=args.debug)
    prepare_snli(tokenizer, args.cache_dir, debug=args.debug)
    prepare_paws(tokenizer, args.cache_dir, debug=args.debug)
    prepare_stsb(tokenizer, args.cache_dir, debug=args.debug)

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f" HOÀN THÀNH CHUẨN BỊ DỮ LIỆU!")
    logger.info(f"   Tổng thời gian: {total_elapsed:.0f}s ({total_elapsed/60:.1f} phút)")
    logger.info(f"   Dữ liệu lưu tại: {args.cache_dir}")
    logger.info("=" * 60)

    # In tóm tắt dung lượng
    total_size = 0
    for root, dirs, files in os.walk(args.cache_dir):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    logger.info(f"   Tổng dung lượng: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
