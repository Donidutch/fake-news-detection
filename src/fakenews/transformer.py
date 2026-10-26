"""Fine-tuning a pretrained transformer encoder for binary classification.

A plain PyTorch loop rather than ``Trainer``: it is short, and every step is visible.
Requires the ``transformer`` extra.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from fakenews.evaluate import metrics


@dataclass
class TrainConfig:
    model_name: str = "roberta-base"
    max_length: int = 256
    batch_size: int = 32
    epochs: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 0


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(texts, labels, tokenizer, max_length, batch_size, shuffle):
    items = list(zip(texts, labels, strict=True))

    def collate(batch):
        enc = tokenizer(
            [t for t, _ in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc

    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_length=256, batch_size=128) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    out = []
    loader = _loader(list(texts), [0] * len(texts), tokenizer, max_length, batch_size, False)
    for batch in loader:
        batch.pop("labels")
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**batch.to(device)).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train(train_df, val_df, cfg: TrainConfig, out_dir: Path) -> dict:
    """Fine-tune on ``train_df``, keep the checkpoint with the best validation macro F1."""
    torch.manual_seed(cfg.seed)
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    model.to(device)

    loader = _loader(
        train_df["content"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        cfg.max_length,
        cfg.batch_size,
        True,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = len(loader) * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, math.ceil(cfg.warmup_ratio * total), total)

    history, best = [], -1.0
    for epoch in range(cfg.epochs):
        model.train()
        start, running = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 500 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(loader)} loss {running / step:.4f}")

        p_val = predict_proba(model, tokenizer, val_df["content"].tolist(), cfg.max_length)
        val = metrics(val_df["label"].to_numpy(), p_val)
        epoch_log = {"epoch": epoch + 1, "train_loss": running / len(loader)}
        epoch_log["seconds"] = time.time() - start
        history.append(epoch_log | {f"val_{k}": v for k, v in val.items()})
        print(json.dumps(history[-1]))
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"best_val_macro_f1": best, "history": history}


def load(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device())
    return model, tokenizer
