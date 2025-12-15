import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ================================
# 1. LOAD DATA
# ================================
FILE = "/Users/aysasorahi/Documents/master/SLAM LAB/REZA/data/LOOK_data.ods"
df = pd.read_excel(FILE, engine="odf")

# Normalize labels
df["function"] = (
    df["function"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Keep only DIR / DM / INTJ
valid = {"DIR", "DM", "INTJ"}
df = df[df["function"].isin(valid)]

# REAL = Label is NaN
df_real = df[df["Label"].isna()].copy()

# SYNTHETIC = Label is not NaN
df_syn = df[df["Label"].notna()].copy()

print("\nREAL counts:")
print(df_real["function"].value_counts())

print("\nSYNTHETIC counts:")
print(df_syn["function"].value_counts())

# ================================
# 2. LABEL MAPPING
# ================================
labels = sorted(valid)
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}

df_real["label_id"] = df_real["function"].map(lab2id)
df_syn["label_id"] = df_syn["function"].map(lab2id)

# ================================
# 3. 50/50 SPLIT ON REAL (TEST ONLY)
# ================================
_, real_test = train_test_split(
    df_real,
    test_size=0.5,
    random_state=42,
    stratify=df_real["label_id"]
)

print("\nREAL TEST (50%) counts:")
print(real_test["function"].value_counts())

# ================================
# 4. DATASET CLASS
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

train_ds = LookDataset(df_syn["sample"], df_syn["label_id"])
test_ds  = LookDataset(real_test["sample"], real_test["label_id"])

# ================================
# 5. MODEL
# ================================
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels),
    id2label=id2lab,
    label2id=lab2id,
)

# ================================
# 6. TRAINING
# ================================
args = TrainingArguments(
    output_dir="syn_to_real_results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="no",
    save_strategy="no",
    logging_steps=50,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

trainer.train()

# ================================
# 7. EVALUATION (REAL TEST)
# ================================
pred = trainer.predict(test_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

print("\n=== SYNTHETIC → REAL (50%) RESULTS ===\n")

print(
    classification_report(
        [id2lab[i] for i in y_true],
        [id2lab[i] for i in y_pred],
        digits=3,
        zero_division=0,
    )
)
