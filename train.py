import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int
    n_classes: int
    model_name: str
    max_length: int
    top_n: int
    max_rows: int | None


def parse_args():
    p = argparse.ArgumentParser(description="Wine variety classification from descriptions (HF Transformers).")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV file (e.g., winemag-data_first150k.csv)")
    p.add_argument("--text_col", type=str, default="description", help="Text column name")
    p.add_argument("--label_col", type=str, default="variety", help="Label column name")
    p.add_argument("--model", type=str, default="distilbert-base-uncased",
                   help="HF model name (distilbert recommended for speed)")
    p.add_argument("--top_n", type=int, default=4, help="Keep top N most frequent classes")
    p.add_argument("--max_rows", type=int, default=None,
                   help="Optional cap on rows AFTER filtering to top_n classes (for fast iteration)")
    p.add_argument("--max_length", type=int, default=128, help="Tokenizer truncation length (128/256 typical)")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--epochs", type=int, default=3, help="Epochs (start with 3; increase after it works)")
    p.add_argument("--batch_size", type=int, default=16, help="Per-device train batch size")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name; defaults to timestamped")
    p.add_argument("--output_dir", type=str, default="outputs", help="Base output directory")
    return p.parse_args()


def load_and_filter(csv_path: str, text_col: str, label_col: str, top_n: int, max_rows: int | None, seed: int):
    # Use only needed columns to reduce memory.
    df = pd.read_csv(csv_path, usecols=[text_col, label_col])

    # Basic cleanup
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    # Keep top N labels
    counts = df[label_col].value_counts()
    keep = counts.head(top_n).index.tolist()
    df = df[df[label_col].isin(keep)].copy()

    # Optional cap for quick iteration (stratified sampling)
    if max_rows is not None and len(df) > max_rows:
        df = (
            df.groupby(label_col, group_keys=False)
              .apply(lambda g: g.sample(n=max_rows // top_n, random_state=seed) if len(g) > max_rows // top_n else g)
        )
        # If rounding left you short, top up randomly (still within filtered set)
        if len(df) < max_rows:
            df = pd.concat([df, df.sample(n=max_rows - len(df), random_state=seed)], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save counts (useful for poster)
    label_counts = df[label_col].value_counts().rename_axis("label").reset_index(name="count")

    return df, label_counts


def make_datasets(df: pd.DataFrame, text_col: str, label_col: str, test_size: float, seed: int):
    labels = sorted(df[label_col].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    y = df[label_col].map(label2id).values
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].values,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
    test_ds = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})
    return train_ds, test_ds, label2id, id2label


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": mf1}


def save_confusion_matrix(y_true, y_pred, id2label, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(id2label.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[id2label[i] for i in id2label.keys()])
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or f"{args.model.replace('/', '_')}_top{args.top_n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load/filter
    df, label_counts = load_and_filter(
        args.csv, args.text_col, args.label_col, args.top_n, args.max_rows, args.seed
    )
    label_counts.to_csv(os.path.join(out_dir, "label_counts.csv"), index=False)

    # Split
    train_ds, test_ds, label2id, id2label = make_datasets(df, args.text_col, args.label_col, args.test_size, args.seed)

    # Tokenizer + dynamic padding collator
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_tok = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    # Mixed precision if CUDA
    fp16 = torch.cuda.is_available()

    train_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "model"),
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # Train
    trainer.train()

    # Evaluate + confusion matrix
    eval_metrics = trainer.evaluate()
    preds = trainer.predict(test_tok)
    y_true = np.array(test_ds["label"])
    y_pred = np.argmax(preds.predictions, axis=-1)

    save_confusion_matrix(
        y_true, y_pred, id2label,
        os.path.join(out_dir, "confusion_matrix.png")
    )

    m = Metrics(
        accuracy=float(eval_metrics["eval_accuracy"]),
        macro_f1=float(eval_metrics["eval_macro_f1"]),
        n_train=len(train_ds),
        n_test=len(test_ds),
        n_classes=len(label2id),
        model_name=args.model,
        max_length=args.max_length,
        top_n=args.top_n,
        max_rows=args.max_rows,
    )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(m), f, indent=2)

    print("Done.")
    print(json.dumps(asdict(m), indent=2))


if __name__ == "__main__":
    main()
