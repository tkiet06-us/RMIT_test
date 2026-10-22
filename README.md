# Multilingual prompt safety classifier

Fine-tune XLM-RoBERTa Base or mDeBERTa-v3-base to classify prompts as `BENIGN`, `JAILBREAK`, or `PROMPT_INJECTION`.

## Project layout

```text
configs/        Reproducible model and training defaults
data/           Input data (the supplied CSVs remain at the project root)
outputs/        Run artifacts, fold checkpoints, metrics, submissions
src/            Data validation, CV, tuning, training, inference
```

## Setup

Use Python 3.10+ and a CUDA-capable PyTorch install when available.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For GPU, install the PyTorch build appropriate for the machine before installing the remaining requirements.

## Train and tune

Run one model at a time; each run uses `AutoModelForSequenceClassification` with full fine-tuning (no LoRA/QLoRA) and outputs are isolated by model/run name.

```powershell
python -m src.train --config configs/xlmr.yaml --trials 10
python -m src.train --config configs/mdeberta.yaml --trials 10
```

Regenerate a submission later from the saved fold checkpoints:

```powershell
python -m src.predict --run-dir outputs/xlmr-base/<run-id>
```

The prediction program reads the run config, predicts with every saved `fold_*/best_model`, averages class probabilities, and writes the predicted class into a copy of `sample_submission.csv`. Optional `--test`, `--sample-submission`, and `--output` arguments override the paths.

Each Optuna trial is scored using stratified 5-fold cross-validation on training data only, stratifying jointly by label and language when every group has enough examples. The best trial is then refit across the same 5 folds; fold checkpoints are retained and test probabilities are averaged across folds. The held-out `test.csv` is never used for tuning or model selection. Metrics include macro F1 (selection metric), weighted F1, accuracy, per-class scores, per-language reports, confusion matrix, and out-of-fold predictions. `sample_submission.csv` is used to preserve submission row order/schema.

Configure fold count, trials, model, seed, batch size, and sequence length in YAML or override with CLI flags. Tune trial count against available compute; each trial trains one model per fold. Shorter max sequence lengths greatly reduce memory and runtime. If VRAM is limited, lower batch size and use gradient accumulation.

## Data contract

Root files: `train.csv` (`id,text,language,label`), `test.csv` (`id,text,language`), and `sample_submission.csv` (`id,label`). Labels are normalized to the required canonical spelling `PROMPT_INJECTION` (the task brief's `PROMPT_INJECTJION` is treated as a typo). Supported labels are validated before training. Language is prepended as a lightweight input hint, e.g. `[LANG=fr] ...`; text is otherwise not translated or altered.

## Reproducibility and leakage controls

- Fixed seed and stratified folds; folds are constructed from training labels only.
- No test-based early stopping, tuning, or threshold selection.
- Macro F1 is used for selecting hyperparameters because classes are imbalanced.
- The five fold models form the final test ensemble; there is no separate test-fit pass.
- Run manifests capture config, package versions, best parameters, fold metrics, and label mapping.

## Artifacts

`outputs/<model>/<run>/` contains `best_params.json`, `cv_metrics.json`, `oof_predictions.csv`, `submission.csv`, and `fold_*/best_model/` checkpoints. Keep large checkpoints out of version control.
