"""Predict test labels from a completed CV run and write a Kaggle submission."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from src.data import LABELS, PromptDataset


def load_test(test_path, sample_path):
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    if not {"id", "text", "language"}.issubset(test.columns):
        raise ValueError("test.csv must contain id,text,language")
    if not {"id", "label"}.issubset(sample.columns):
        raise ValueError("sample_submission.csv must contain id,label")
    if test[["id", "text", "language"]].isna().any().any():
        raise ValueError("Missing id, text, or language in test data")
    if len(sample) != len(test) or not sample["id"].equals(test["id"]):
        raise ValueError("sample_submission IDs/order must exactly match test.csv")
    test["model_text"] = test.apply(lambda row: f"[LANG={row.language}] {row.text}", axis=1)
    return test, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed outputs/<model>/<run> directory")
    parser.add_argument("--test", help="Override test CSV path; defaults to the path recorded in config.yaml")
    parser.add_argument("--sample-submission", help="Override sample submission path")
    parser.add_argument("--output", help="Output CSV; defaults to <run-dir>/submission.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    test_path = args.test or cfg["test_path"]
    sample_path = args.sample_submission or cfg["sample_submission_path"]
    test, result = load_test(test_path, sample_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = PromptDataset(test.model_text, tokenizer, cfg["max_length"])
    collator = DataCollatorWithPadding(tokenizer)
    fold_dirs = sorted(run_dir.glob("fold_*/best_model"), key=lambda p: int(p.parent.name.split("_")[-1]))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold checkpoints found under {run_dir}")

    probabilities = []
    for fold_dir in fold_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(fold_dir)
        predict_args = TrainingArguments(
            output_dir=str(run_dir / "predict_tmp"),
            per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 16),
            report_to="none",
        )
        trainer = Trainer(model=model, args=predict_args, data_collator=collator)
        logits = trainer.predict(dataset).predictions
        probabilities.append(torch.softmax(torch.as_tensor(logits), dim=-1).cpu().numpy())
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_probs = np.mean(probabilities, axis=0)
    result["label"] = [LABELS[i] for i in mean_probs.argmax(axis=1)]
    output = Path(args.output) if args.output else run_dir / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} predictions to {output}")


if __name__ == "__main__":
    main()
