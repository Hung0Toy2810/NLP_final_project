import os
import sys
from typing import Any, cast

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, get_device
from dataset import STSBDataset, create_dataloader, get_tokenizer
from model.encoder import create_sftbe_model


def evaluate_stsb(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)

            emb_a = model(ids_a, mask_a)
            emb_b = model(ids_b, mask_b)
            cosine = F.cosine_similarity(emb_a, emb_b, dim=-1)
            predictions.extend(cosine.cpu().tolist())
            labels.extend(batch["score"].tolist())

    correlation = spearmanr(predictions, labels)[0]
    return float(cast(Any, correlation))


def main():
    device = get_device()
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    dataset = STSBDataset(
        TRAIN_CONFIG["data_cache_dir"],
        tokenizer,
        split="test",
        max_length=MODEL_CONFIG["max_seq_length"],
    )
    loader = create_dataloader(
        dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = create_sftbe_model(MODEL_CONFIG).to(device)
    checkpoint = os.environ.get(
        "SFTBE_EVAL_MODEL_PATH",
        os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage0_final.pt"),
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    score = evaluate_stsb(model, loader, device)
    print(f"STS-B Spearman: {score:.4f}")


if __name__ == "__main__":
    main()
