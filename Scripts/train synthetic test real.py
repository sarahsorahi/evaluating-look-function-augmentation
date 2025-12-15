import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

# ================================
# 1. LOAD DATA
# ================================
FILE = "/Users/aysasorahi/Documents/master/SLAM LAB/REZA/data/LOOK_data.ods"
df = pd.read_excel(FILE, engine="odf")

df["function"] = (
    df["function"]
    .astype(str)
    .str.replace(r"\s+", "", regex=True)
    .str.strip()
    .str.upper()
)

# Keep only DIR / DM / INTJ (NO AS)
valid = {"DIR", "DM", "INTJ"}
df = df[df["function"].isin(valid)].copy()

df_real = df[df["Label"].isna()].copy()
df_syn  = df[df["Label"].notna()].copy()

print("\nREAL counts:")
print(df_real["function"].value_counts())

print("\nSYNTHETIC counts:")
print(df_syn["function"].value_counts())

# ================================
# 2. LABELS
# ================================
labels = sorted(list(valid))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}

df_real["label_id"] = df_real["function"].map(lab2id)
df_syn["label_id"]  = df_syn["function"].map(lab2id)

# ================================
# 3. DATASET
# ================================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class LookDataset(Dataset):
    def __init__(self, texts, labels):
        enc = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=128,
        )
        self.enc = {k: torch.tensor(v) for k, v in enc.items()}
        self.labels = torch.tensor(labels.tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item

train_ds = LookDataset(df_syn["sample"].astype(str), df_syn["label_id"])
test_ds  = LookDataset(df_real["sample"].astype(str), df_real["label_id"])

# ================================
# 4. MODEL
# ================================
set_seed(42)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels),
    id2label=id2lab,
    label2id=lab2id,
)

# ================================
# 5. TRAINING ARGS (FIXED)
# ================================
args = TrainingArguments(
    output_dir="synthetic_to_real",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="no",   # <-- FIX: use eval_strategy (not evaluation_strategy)
    save_strategy="no",
    logging_steps=50,
    seed=42,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

# ================================
# 6. TRAIN
# ================================
trainer.train()

# ================================
# 7. EVAL ON REAL
# ================================
pred = trainer.predict(test_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

print("\n=== TRAIN: SYNTHETIC → TEST: REAL (DIR/DM/INTJ) ===\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=labels,
        digits=3,
        zero_division=0,
    )
)
