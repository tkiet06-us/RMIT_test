import argparse
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import sklearn
import torch
import transformers
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

from src.data import LABELS, LABEL2ID, PromptDataset, load_data


def metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, pred, average="macro"),
            "weighted_f1": f1_score(labels, pred, average="weighted"),
            "accuracy": accuracy_score(labels, pred)}


def make_trainer(cfg, trial_params, train_ds, val_ds, tokenizer, out, trial_tag):
    args = TrainingArguments(
        output_dir=str(out), learning_rate=trial_params["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=trial_params["batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        weight_decay=trial_params["weight_decay"], warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=trial_params["lr_scheduler_type"],
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps",
        logging_steps=100, load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1, report_to="none",
        seed=cfg["seed"], data_seed=cfg["seed"], fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", False),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS), id2label={i: x for i, x in enumerate(LABELS)},
        label2id=LABEL2ID,
    )
    return Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                   tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                   compute_metrics=metrics,
                   callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--folds", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.folds:
        cfg["n_splits"] = args.folds
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    train, test, sample = load_data(cfg["train_path"], cfg["test_path"], cfg["sample_submission_path"])
    min_class = int(train.label_id.value_counts().min())
    if cfg["n_splits"] > min_class:
        raise ValueError(f"n_splits={cfg['n_splits']} exceeds smallest class count {min_class}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_ds = PromptDataset(train.model_text, tokenizer, cfg["max_length"], train.label_id)
    test_ds = PromptDataset(test.model_text, tokenizer, cfg["max_length"])
    y = train.label_id.to_numpy()
    splitter = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    joint_strata = train["label"].astype(str) + "::" + train["language"].astype(str)
    if int(joint_strata.value_counts().min()) >= cfg["n_splits"]:
        strata = joint_strata
        stratification = "label_and_language"
    else:
        strata = y
        stratification = "label (fallback: a label-language group is smaller than n_splits)"
    splits = list(splitter.split(np.zeros(len(y)), strata))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(cfg["output_dir"]) / cfg["run_name"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 8e-6, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
        }
        fold_scores = []
        for fold, (tr, va) in enumerate(splits):
            set_seed(cfg["seed"] + fold)
            trainer = make_trainer(cfg, params, torch.utils.data.Subset(train_ds, tr),
                                   torch.utils.data.Subset(train_ds, va), tokenizer,
                                   out / "tuning" / f"trial_{trial.number}" / f"fold_{fold}", trial.number)
            trainer.train()
            fold_scores.append(float(trainer.evaluate()["eval_macro_f1"]))
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        trial.set_user_attr("fold_macro_f1", fold_scores)
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", study_name=cfg["run_name"])
    study.optimize(objective, n_trials=args.trials)
    best = study.best_params
    shutil.rmtree(out / "tuning", ignore_errors=True)
    (out / "best_params.json").write_text(json.dumps({"params": best, "cv_macro_f1": study.best_value}, indent=2), encoding="utf-8")
    pd.DataFrame([{"trial": t.number, "value": t.value, **t.params,
                   "fold_macro_f1": t.user_attrs.get("fold_macro_f1")} for t in study.trials]).to_json(
                       out / "tuning_trials.json", orient="records", indent=2)

    oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    test_probs = []
    fold_results = []
    for fold, (tr, va) in enumerate(splits):
        set_seed(cfg["seed"] + fold)
        trainer = make_trainer(cfg, best, torch.utils.data.Subset(train_ds, tr),
                               torch.utils.data.Subset(train_ds, va), tokenizer,
                               out / f"fold_{fold}", f"final_{fold}")
        trainer.train()
        val_out = trainer.predict(torch.utils.data.Subset(train_ds, va))
        probs = torch.softmax(torch.tensor(val_out.predictions), dim=-1).numpy()
        oof[va] = probs
        fold_results.append({"fold": fold, "macro_f1": f1_score(y[va], probs.argmax(1), average="macro"),
                             "weighted_f1": f1_score(y[va], probs.argmax(1), average="weighted"),
                             "accuracy": accuracy_score(y[va], probs.argmax(1))})
        trainer.save_model(out / f"fold_{fold}" / "best_model")
        test_out = trainer.predict(test_ds)
        test_probs.append(torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy())
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    oof_pred = oof.argmax(axis=1)
    summary = {"folds": fold_results,
               "stratification": stratification,
               "oof_macro_f1": f1_score(y, oof_pred, average="macro"),
               "oof_weighted_f1": f1_score(y, oof_pred, average="weighted"),
               "oof_accuracy": accuracy_score(y, oof_pred),
               "classification_report": classification_report(y, oof_pred, labels=range(len(LABELS)), target_names=LABELS, output_dict=True),
               "confusion_matrix": confusion_matrix(y, oof_pred, labels=range(len(LABELS))).tolist(),
               "per_language": {
                   str(language): {
                       "count": int(mask.sum()),
                       "macro_f1": f1_score(y[mask], oof_pred[mask], average="macro", labels=range(len(LABELS)), zero_division=0),
                       "classification_report": classification_report(
                           y[mask], oof_pred[mask], labels=range(len(LABELS)), target_names=LABELS,
                           output_dict=True, zero_division=0),
                   }
                   for language in sorted(train.language.astype(str).unique())
                   for mask in [train.language.astype(str).to_numpy() == language]
               },
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "transformers": transformers.__version__, "sklearn": sklearn.__version__}}
    (out / "cv_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"id": train.id, "label": train.label,
                  **{f"prob_{lab}": oof[:, i] for i, lab in enumerate(LABELS)},
                  "oof_prediction": [LABELS[i] for i in oof_pred]}).to_csv(out / "oof_predictions.csv", index=False)
    final_probs = np.mean(test_probs, axis=0)
    result = sample.copy()
    result["label"] = [LABELS[i] for i in final_probs.argmax(axis=1)]
    result.to_csv(out / "submission.csv", index=False)
    print(f"Completed. CV macro F1={summary['oof_macro_f1']:.4f}; artifacts: {out}")


if __name__ == "__main__":
    main()
