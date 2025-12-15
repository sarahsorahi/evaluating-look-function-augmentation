import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ======================================
# 1. LOAD & FILTER REAL DATA ONLY
# ======================================
FILE = "Data/LOOK_data.ods"   # <-- adjust if needed
df = pd.read_excel(FILE, engine="odf")

# REAL data = Label is NaN
df = df[df["Label"].isna()].copy()

# Normalize function labels
df["function"] = (
    df["function"]
    .astype(str)
    .str.replace(r"\s+", "", regex=True)
    .str.strip()
    .str.upper()
)

VALID = {"AS", "DIR", "DM", "INTJ"}
df = df[df["function"].isin(VALID)].reset_index(drop=True)

print("\nREAL data distribution:")
print(df["function"].value_counts())

# Safety check
counts = df["function"].value_counts()
if (counts < 2).any():
    raise ValueError("Each class must have at least 2 real samples.")

# ======================================
# 2. LABEL MAPPING
# ======================================
labels = sorted(VALID)
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}

df["label_id"] = df["function"].map(lab2id)

X = df["sample"].astype(str).values
y = df["label_id"].values

# ======================================
# 3. DATASET
# ======================================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class LookDataset(Dataset):
    def __init__(self, texts, labels):
        enc = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=128,
        )
        self.encodings = {k: torch.tensor(v) for k, v in enc.items()}
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# ======================================
# 4. REPEATED 2-FOLD CV (50/50)
# ======================================
N_RUNS = 5
N_SPLITS = 2

acc_scores = []
macro_f1_scores = []

all_true = []
all_pred = []

for run in range(N_RUNS):
    seed = 42 + run
    set_seed(seed)

    print(f"\n========== RUN {run+1}/{N_RUNS} (seed={seed}) ==========")

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")

        train_ds = LookDataset(X[train_idx], y[train_idx])
        test_ds  = LookDataset(X[test_idx],  y[test_idx])

        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(labels),
            id2label=id2lab,
            label2id=lab2id,
        )

        args = TrainingArguments(
            output_dir=f"results/real_only/run{run+1}_fold{fold}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_strategy="no",
            eval_strategy="no",
            logging_steps=50,
            seed=seed,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
        )

        trainer.train()

        preds = trainer.predict(test_ds)
        y_true = preds.label_ids
        y_hat = np.argmax(preds.predictions, axis=1)

        acc = accuracy_score(y_true, y_hat)
        macro_f1 = f1_score(y_true, y_hat, average="macro")

        acc_scores.append(acc)
        macro_f1_scores.append(macro_f1)

        all_true.extend(y_true.tolist())
        all_pred.extend(y_hat.tolist())

        print(f"Fold Accuracy: {acc:.3f}")
        print(f"Fold Macro F1: {macro_f1:.3f}")

# ======================================
# 5. FINAL SUMMARY
# ======================================
print("\n==============================")
print("FINAL RESULTS (REAL-ONLY)")
print(f"Evaluations: {N_RUNS * N_SPLITS}")
print(f"Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
print(f"Macro F1:  {np.mean(macro_f1_scores):.3f} ± {np.std(macro_f1_scores):.3f}")

print("\nAggregated Classification Report:")
print(
    classification_report(
        [id2lab[i] for i in all_true],
        [id2lab[i] for i in all_pred],
        digits=3,
        zero_division=0,
    )
)
