import pandas as pd
import torch

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ALIASES = {"PROMPT_INJECTJION": "PROMPT_INJECTION"}


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = None if labels is None else torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx].item()
        return item


def load_data(train_path, test_path, sample_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    required_train = {"id", "text", "language", "label"}
    required_test = {"id", "text", "language"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"train.csv must contain {sorted(required_train)}")
    if not required_test.issubset(test.columns):
        raise ValueError(f"test.csv must contain {sorted(required_test)}")
    if not {"id", "label"}.issubset(sample.columns):
        raise ValueError("sample_submission.csv must contain id,label")
    if train[["id", "text", "language", "label"]].isna().any().any():
        raise ValueError("Missing id, text, language, or label in training data")
    if test[["id", "text", "language"]].isna().any().any():
        raise ValueError("Missing id, text, or language in test data")
    train["label"] = train["label"].astype(str).str.strip().str.upper().replace(ALIASES)
    unknown = sorted(set(train.label) - set(LABELS))
    if unknown:
        raise ValueError(f"Unsupported labels: {unknown}; expected {LABELS}")
    if not test.id.is_unique or not train.id.is_unique:
        raise ValueError("IDs must be unique")
    if len(sample) != len(test) or not sample.id.equals(test.id):
        raise ValueError("sample_submission IDs/order must exactly match test.csv")
    train["label_id"] = train.label.map(LABEL2ID).astype(int)
    train["model_text"] = train.apply(lambda r: f"[LANG={r.language}] {r.text}", axis=1)
    test["model_text"] = test.apply(lambda r: f"[LANG={r.language}] {r.text}", axis=1)
    return train, test, sample
