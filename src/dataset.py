# =============================================================================
# dataset.py — Custom Dataset & DataLoader (PyTorch thuần)
# =============================================================================
# Hỗ trợ 2 chế độ:
#   (A) Pre-tokenized mode: Đọc từ cache Apache Arrow (cho H100 — tốc độ tối đa)
#   (B) On-the-fly mode: Tokenize trong __getitem__ (cho debug MPS — thuận tiện)
#
# Bài báo tham khảo (Datasets):
#   [1] Bowman et al., "A large annotated corpus for learning NLI", EMNLP 2015 → SNLI
#   [2] Williams et al., "A Broad-Coverage Challenge Corpus for NLI", NAACL 2018 → MultiNLI
#   [3] Zhang et al., "PAWS: Paraphrase Adversaries from Word Scrambling", NAACL 2019
#   [4] Cer et al., "STS Benchmark", SemEval@ACL 2017
#   [5] Gao et al., "SimCSE: Simple Contrastive Learning of Sentence Embeddings", EMNLP 2021
#
# Chiến lược dữ liệu:
#   [6] Bengio et al., "Curriculum Learning", ICML 2009
#       → Stage 0 (SimCSE, dễ nhất) → Stage 1 (NLI, dễ) → Stage 2 (Similarity + HN, khó)
# =============================================================================

import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast  # CHỈ dùng tokenizer, không dùng model
from datasets import load_dataset, load_from_disk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# TOKENIZER — Devlin et al., "BERT", NAACL 2019
# Chỉ dùng BertTokenizerFast để tách chuỗi text thành Token IDs.
# Toàn bộ neural network phía sau là PyTorch thuần.
# =============================================================================

def get_tokenizer(tokenizer_name: str = "bert-base-uncased"):
    """Load WordPiece tokenizer từ HuggingFace (chỉ tokenizer, không model)."""
    return BertTokenizerFast.from_pretrained(tokenizer_name)


# =============================================================================
# DATASET CHO GIAI ĐOẠN 0: UNSUPERVISED SimCSE (Wikipedia)
# =============================================================================
# Gao et al., "SimCSE", EMNLP 2021:
#   "Unsupervised SimCSE simply takes an input sentence and predicts itself
#    in a contrastive objective, with only standard dropout used as noise."
#
# Cách hoạt động:
#   1. Mỗi sample chỉ có 1 câu (không cần cặp câu hay nhãn)
#   2. Trong training loop, cùng 1 câu được encode 2 lần
#   3. Do nn.Dropout(0.1) khác nhau mỗi lần forward → tạo ra 2 vector khác nhau
#   4. MNR Loss ép 2 vector (cùng 1 câu) lại gần nhau, đẩy câu khác ra xa

class WikipediaSimCSEDataset(Dataset):
    """
    Dataset cho Stage 0 — Unsupervised SimCSE trên Wikipedia.
    Gao et al., EMNLP 2021.

    Mỗi sample chỉ trả về input_ids và attention_mask của 1 câu.
    Positive pair được tạo tự động nhờ Dropout trong training loop.
    """

    def __init__(self, cache_dir: str = None, tokenizer=None,
                 max_length: int = 128, debug: bool = True,
                 num_debug_samples: int = 5000):
        self.max_length = max_length
        self.use_cache = cache_dir is not None and os.path.exists(
            os.path.join(cache_dir, "wikipedia_tokenized")
        )

        if self.use_cache:
            # Chế độ A: Đọc từ cache pre-tokenized (Apache Arrow, zero-copy)
            cache_path = os.path.join(cache_dir, "wikipedia_tokenized")
            logger.info(f"[Cache] Loading pre-tokenized Wikipedia: {cache_path}")
            self.data = load_from_disk(cache_path)
            if debug:
                self.data = self.data.select(range(min(num_debug_samples, len(self.data))))
            logger.info(f"Wikipedia sentences loaded: {len(self.data):,}")
        else:
            # Chế độ B: Tải on-the-fly (cho debug MPS khi chưa chạy prepare_data.py)
            logger.info("[On-the-fly] Tải Wikipedia trực tiếp (streaming debug)...")
            self.tokenizer = tokenizer
            wiki = load_dataset(
                "wikimedia/wikipedia", "20231101.en",
                split="train",
                streaming=True
            ).take(num_debug_samples)
            # Tách thành câu
            sentences = []
            for article in wiki:
                for sent in article['text'].split('. '):
                    sent = sent.strip()
                    if 20 <= len(sent) <= 500:
                        sentences.append(sent)
                        if len(sentences) >= num_debug_samples:
                            break
                if len(sentences) >= num_debug_samples:
                    break
            self.sentences = sentences[:num_debug_samples]
            logger.info(f"[DEBUG] {len(self.sentences)} sentences from Wikipedia")

    def __len__(self):
        if self.use_cache:
            return len(self.data)
        return len(self.sentences)

    def __getitem__(self, idx):
        if self.use_cache:
            item = self.data[idx]
            return {
                'input_ids': item['input_ids'].clone().detach().long(),
                'attention_mask': item['attention_mask'].clone().detach().long(),
            }
        else:
            enc = self.tokenizer(
                self.sentences[idx],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
            }


# =============================================================================
# DATASET CHO GIAI ĐOẠN 1: NLI (Natural Language Inference)
# =============================================================================
# Bowman et al., EMNLP 2015 (SNLI)
# 3 classes: Entailment (0), Neutral (1), Contradiction (2)

class NLIDataset(Dataset):
    """
    Dataset cho giai đoạn NLI — Curriculum Learning Stage 1 (dễ).
    Bengio et al., ICML 2009: bắt đầu từ task dễ (phân loại 3 classes).
    """

    def __init__(self, cache_dir: str = None, tokenizer=None,
                 max_length: int = 128, debug: bool = True,
                 num_debug_samples: int = 3000):
        self.max_length = max_length
        self.use_cache = cache_dir is not None and os.path.exists(
            os.path.join(cache_dir, "snli_tokenized")
        )

        if self.use_cache:
            cache_path = os.path.join(cache_dir, "snli_tokenized")
            logger.info(f"[Cache] Loading pre-tokenized SNLI: {cache_path}")
            self.data = load_from_disk(cache_path)['train']
            if debug:
                self.data = self.data.select(range(min(num_debug_samples, len(self.data))))
            logger.info(f"SNLI samples loaded: {len(self.data):,}")
        else:
            self.tokenizer = tokenizer
            logger.info("Đang tải SNLI dataset (Bowman et al., EMNLP 2015)...")
            dataset = load_dataset("stanfordnlp/snli", split="train")
            dataset = dataset.filter(lambda x: x['label'] != -1)
            if debug:
                dataset = dataset.select(range(min(num_debug_samples, len(dataset))))
                logger.info(f"[DEBUG] Sử dụng {len(dataset)} samples NLI")
            self.raw_data = dataset

    def __len__(self):
        if self.use_cache:
            return len(self.data)
        return len(self.raw_data)

    def __getitem__(self, idx):
        if self.use_cache:
            item = self.data[idx]
            return {
                'input_ids_a': item['input_ids_a'].clone().detach().long(),
                'attention_mask_a': item['attention_mask_a'].clone().detach().long(),
                'input_ids_b': item['input_ids_b'].clone().detach().long(),
                'attention_mask_b': item['attention_mask_b'].clone().detach().long(),
                'label': item['label'].clone().detach().long() if isinstance(item['label'], torch.Tensor) else torch.tensor(item['label'], dtype=torch.long),
            }
        else:
            item = self.raw_data[idx]
            enc1 = self.tokenizer(item['premise'], max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
            enc2 = self.tokenizer(item['hypothesis'], max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids_a': enc1['input_ids'].squeeze(0),
                'attention_mask_a': enc1['attention_mask'].squeeze(0),
                'input_ids_b': enc2['input_ids'].squeeze(0),
                'attention_mask_b': enc2['attention_mask'].squeeze(0),
                'label': torch.tensor(item['label'], dtype=torch.long)
            }


# =============================================================================
# DATASET CHO GIAI ĐOẠN 2: SIMILARITY (Contrastive Learning + Hard Negatives)
# =============================================================================
# Dùng PAWS — Zhang et al., NAACL 2019

class SimilarityDataset(Dataset):
    """
    Dataset cho giai đoạn Similarity — Curriculum Learning Stage 2 (khó hơn).
    PAWS chứa adversarial hard negatives: câu có cùng từ vựng nhưng khác nghĩa.
    """

    def __init__(self, cache_dir: str = None, tokenizer=None,
                 max_length: int = 128, debug: bool = True,
                 num_debug_samples: int = 3000, include_hard_negatives: bool = False):
        self.max_length = max_length
        self.include_hard_negatives = include_hard_negatives
        self.use_cache = cache_dir is not None and os.path.exists(
            os.path.join(cache_dir, "paws_tokenized")
        )

        if self.use_cache:
            cache_path = os.path.join(cache_dir, "paws_tokenized")
            logger.info(f"[Cache] Loading pre-tokenized PAWS: {cache_path}")
            full_data = load_from_disk(cache_path)['train']

            if include_hard_negatives:
                self.positives = full_data.filter(lambda x: x['label'] == 1)
                self.negatives = full_data.filter(lambda x: x['label'] == 0)
                if debug:
                    limit = min(num_debug_samples // 2, len(self.positives), len(self.negatives))
                    self.positives = self.positives.select(range(limit))
                    self.negatives = self.negatives.select(range(limit))
                logger.info(f"PAWS: {len(self.positives)} pos + {len(self.negatives)} neg")
            else:
                self.data = full_data.filter(lambda x: x['label'] == 1)
                if debug:
                    self.data = self.data.select(range(min(num_debug_samples, len(self.data))))
                logger.info(f"PAWS positive pairs: {len(self.data):,}")
        else:
            self.tokenizer = tokenizer
            logger.info("Đang tải PAWS dataset (Zhang et al., NAACL 2019)...")
            dataset = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
            if include_hard_negatives:
                self.positives = dataset.filter(lambda x: x['label'] == 1)
                self.negatives = dataset.filter(lambda x: x['label'] == 0)
                if debug:
                    limit = min(num_debug_samples // 2, len(self.positives), len(self.negatives))
                    self.positives = self.positives.select(range(limit))
                    self.negatives = self.negatives.select(range(limit))
            else:
                dataset = dataset.filter(lambda x: x['label'] == 1)
                if debug:
                    dataset = dataset.select(range(min(num_debug_samples, len(dataset))))
                self.raw_data = dataset

    def __len__(self):
        if self.include_hard_negatives:
            return len(self.positives)
        if self.use_cache:
            return len(self.data)
        return len(self.raw_data)

    def _get_pair_from_item(self, item, is_cached):
        """Helper: extract pair tensors from either cached or raw item."""
        if is_cached:
            return {
                'input_ids_a': torch.tensor(item['input_ids_a'], dtype=torch.long),
                'attention_mask_a': torch.tensor(item['attention_mask_a'], dtype=torch.long),
                'input_ids_b': torch.tensor(item['input_ids_b'], dtype=torch.long),
                'attention_mask_b': torch.tensor(item['attention_mask_b'], dtype=torch.long),
            }
        else:
            enc1 = self.tokenizer(item['sentence1'], max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
            enc2 = self.tokenizer(item['sentence2'], max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids_a': enc1['input_ids'].squeeze(0),
                'attention_mask_a': enc1['attention_mask'].squeeze(0),
                'input_ids_b': enc2['input_ids'].squeeze(0),
                'attention_mask_b': enc2['attention_mask'].squeeze(0),
            }

    def __getitem__(self, idx):
        if self.include_hard_negatives:
            pos_item = self.positives[idx]
            neg_idx = idx % len(self.negatives)
            neg_item = self.negatives[neg_idx]

            result = self._get_pair_from_item(pos_item, self.use_cache)

            if self.use_cache:
                result['input_ids_neg'] = torch.tensor(neg_item['input_ids_b'], dtype=torch.long)
                result['attention_mask_neg'] = torch.tensor(neg_item['attention_mask_b'], dtype=torch.long)
            else:
                enc_neg = self.tokenizer(neg_item['sentence2'], max_length=self.max_length,
                                         padding='max_length', truncation=True, return_tensors='pt')
                result['input_ids_neg'] = enc_neg['input_ids'].squeeze(0)
                result['attention_mask_neg'] = enc_neg['attention_mask'].squeeze(0)
            return result
        else:
            if self.use_cache:
                return self._get_pair_from_item(self.data[idx], True)
            else:
                return self._get_pair_from_item(self.raw_data[idx], False)


# =============================================================================
# DATASET CHO ĐÁNH GIÁ (Evaluation)
# =============================================================================

class STSBDataset(Dataset):
    """
    STS Benchmark — Cer et al., SemEval@ACL 2017.
    Đánh giá chất lượng embedding bằng Spearman correlation.
    """

    def __init__(self, cache_dir: str = None, tokenizer=None,
                 max_length: int = 128, split: str = "test",
                 debug: bool = True, num_debug_samples: int = 500):
        self.max_length = max_length
        self.use_cache = cache_dir is not None and os.path.exists(
            os.path.join(cache_dir, "stsb_tokenized")
        )

        if self.use_cache:
            cache_path = os.path.join(cache_dir, "stsb_tokenized")
            logger.info(f"[Cache] Loading pre-tokenized STS-B ({split}): {cache_path}")
            self.data = load_from_disk(cache_path)[split]
            if debug:
                self.data = self.data.select(range(min(num_debug_samples, len(self.data))))
            logger.info(f"STS-B ({split}) loaded: {len(self.data)}")
        else:
            self.tokenizer = tokenizer
            logger.info(f"Đang tải STS-B dataset [{split}]...")
            dataset = load_dataset("mteb/stsbenchmark-sts", split=split)
            if debug:
                dataset = dataset.select(range(min(num_debug_samples, len(dataset))))
            self.raw_data = dataset

    def __len__(self):
        if self.use_cache:
            return len(self.data)
        return len(self.raw_data)

    def __getitem__(self, idx):
        if self.use_cache:
            item = self.data[idx]
            return {
                'input_ids_a': item['input_ids_a'].clone().detach().long(),
                'attention_mask_a': item['attention_mask_a'].clone().detach().long(),
                'input_ids_b': item['input_ids_b'].clone().detach().long(),
                'attention_mask_b': item['attention_mask_b'].clone().detach().long(),
                'score': item['score'].clone().detach().float() if isinstance(item['score'], torch.Tensor) else torch.tensor(item['score'], dtype=torch.float32),
            }
        else:
            item = self.raw_data[idx]
            enc1 = self.tokenizer(item['sentence1'], max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
            enc2 = self.tokenizer(item['sentence2'], max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids_a': enc1['input_ids'].squeeze(0),
                'attention_mask_a': enc1['attention_mask'].squeeze(0),
                'input_ids_b': enc2['input_ids'].squeeze(0),
                'attention_mask_b': enc2['attention_mask'].squeeze(0),
                'score': torch.tensor(item['score'] / 5.0, dtype=torch.float),
            }


class PAWSEvalDataset(Dataset):
    """
    PAWS Evaluation — Zhang et al., NAACL 2019.
    Binary: 0 (not paraphrase), 1 (paraphrase).
    """

    def __init__(self, cache_dir: str = None, tokenizer=None,
                 max_length: int = 128, debug: bool = True,
                 num_debug_samples: int = 500):
        self.max_length = max_length
        self.use_cache = cache_dir is not None and os.path.exists(
            os.path.join(cache_dir, "paws_tokenized")
        )

        if self.use_cache:
            cache_path = os.path.join(cache_dir, "paws_tokenized")
            logger.info(f"[Cache] Loading pre-tokenized PAWS (validation): {cache_path}")
            self.data = load_from_disk(cache_path)['validation']
            if debug:
                self.data = self.data.select(range(min(num_debug_samples, len(self.data))))
        else:
            self.tokenizer = tokenizer
            logger.info("Đang tải PAWS validation...")
            dataset = load_dataset("google-research-datasets/paws", "labeled_final", split="validation")
            if debug:
                dataset = dataset.select(range(min(num_debug_samples, len(dataset))))
            self.raw_data = dataset

    def __len__(self):
        if self.use_cache:
            return len(self.data)
        return len(self.raw_data)

    def __getitem__(self, idx):
        if self.use_cache:
            item = self.data[idx]
            return {
                'input_ids_a': item['input_ids_a'].clone().detach().long(),
                'attention_mask_a': item['attention_mask_a'].clone().detach().long(),
                'input_ids_b': item['input_ids_b'].clone().detach().long(),
                'attention_mask_b': item['attention_mask_b'].clone().detach().long(),
                'label': item['label'].clone().detach().long() if isinstance(item['label'], torch.Tensor) else torch.tensor(item['label'], dtype=torch.long),
            }
        else:
            item = self.raw_data[idx]
            enc1 = self.tokenizer(item['sentence1'], max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
            enc2 = self.tokenizer(item['sentence2'], max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids_a': enc1['input_ids'].squeeze(0),
                'attention_mask_a': enc1['attention_mask'].squeeze(0),
                'input_ids_b': enc2['input_ids'].squeeze(0),
                'attention_mask_b': enc2['attention_mask'].squeeze(0),
                'label': torch.tensor(item['label'], dtype=torch.long),
            }


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                      num_workers: int = 0, prefetch_factor: int = None) -> DataLoader:
    """
    Tạo DataLoader tối ưu cho cả MPS debug và CUDA training.

    Trên CUDA (H100):
        - num_workers=4-8: Đa luồng CPU chuẩn bị batch song song
        - pin_memory=True: Page-locked RAM → copy PCIe nhanh 2x
        - prefetch_factor=2: Luôn chuẩn bị sẵn 2 batch trước
    """
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True,  # Quan trọng cho MNR Loss (cần batch đều)
    }
    # prefetch_factor chỉ hợp lệ khi num_workers > 0
    if num_workers > 0 and prefetch_factor:
        kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(dataset, **kwargs)
