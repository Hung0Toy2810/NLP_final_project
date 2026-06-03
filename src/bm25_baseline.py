# =============================================================================
# bm25_baseline.py — BM25 Baseline (Thí nghiệm 0)
# =============================================================================
# Bài báo tham khảo:
#   Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond",
#   Foundations and Trends in Information Retrieval, 2009.
#
# Mục đích: Chứng minh sự giới hạn của phương pháp lexical matching truyền thống.
# PAWS chứa các câu có từ vựng giống hệt nhau nhưng khác nghĩa → BM25 bị lừa.
# Kết quả mong đợi: Accuracy ~50% (tương đương random guessing).
# =============================================================================

import logging
import numpy as np
from typing import cast
from rank_bm25 import BM25Okapi  # type: ignore
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_bm25_on_paws(debug=True):
    """
    Đánh giá BM25 baseline trên tập PAWS.
    PAWS chứa các câu có lexical overlap cao nhưng trái ngược ngữ nghĩa.
    BM25 dựa trên từ vựng, do đó sẽ dự đoán sai rất nhiều trên tập này.
    Thí nghiệm này nhằm chứng minh sự giới hạn của phương pháp truyền thống.
    """
    logger.info("Đang tải dữ liệu PAWS cho baseline BM25...")
    paws = cast(
        Dataset,
        load_dataset("google-research-datasets/paws", "labeled_final", split="validation")
    )

    y_true = []
    bm25_scores = []

    limit = 500 if debug else len(paws)
    for i, row in enumerate(paws):
        if i >= limit:
            break

        item = cast(dict[str, object], row)
        s1 = str(item['sentence1']).lower().split()
        s2 = str(item['sentence2']).lower().split()

        # Tạo corpus với duy nhất 1 câu (sentence1)
        bm25 = BM25Okapi([s1])
        # Lấy score của sentence2 dựa trên corpus (sentence1)
        score = bm25.get_scores(s2)[0]

        bm25_scores.append(score)
        y_true.append(int(str(item['label'])))

    # BM25 không cho ra xác suất 0-1, ta tìm một ngưỡng (threshold) để chia nhãn
    threshold = np.median(bm25_scores)
    y_pred = [1 if score >= threshold else 0 for score in bm25_scores]

    acc = accuracy_score(y_true, y_pred)
    logger.info(f" BM25 trên PAWS: Accuracy = {acc:.4f}")
    logger.info("Lưu ý: PAWS là tập Adversarial, Acc của BM25 thường rất thấp (~50%).")
    return acc


if __name__ == '__main__':
    try:
        evaluate_bm25_on_paws(debug=True)
    except ModuleNotFoundError:
        logger.error("Thư viện 'rank_bm25' chưa được cài đặt. Chạy: pip install rank_bm25")
