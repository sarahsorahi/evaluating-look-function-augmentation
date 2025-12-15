import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# ================================
# 1. LOAD & CLEAN REAL DATA
# ================================
FILE = "/Users/aysasorahi/Documents/master/SLAM LAB/REZA/data/LOOK_three.ods"
df = pd.read_excel(FILE, engine="odf")

# Keep only REAL data (Label is NaN)
df_real = df[df["Label"].isna()].copy()

# Normalize function names
df_real["function"] = (
    df_real["function"]
    .astype(str)
    .str.replace(r"\s+", "", regex=True)  # remove invisible spaces
    .str.strip()
    .str.upper()
)

# Only keep valid four classes
valid = {"AS", "DIR", "DM", "INTJ"}
df_real = df_real[df_real["function"].isin(valid)]

print("\nCleaned REAL data counts:")
print(df_real["function"].value_counts())


# ================================
# 2. VERIFY MINIMUM CLASS SIZE
# ================================
counts = df_real["function"].value_counts()
too_small = counts[counts < 2]

if len(too_small) > 0:
    print("\n❌ ERROR: Some classes have fewer than 2 samples:")
    print(too_small)
    print("\nPlease fix your data first.")
    exit()

# ================================
# 3. MAP LABELS
# ================================
labels = sorted(df_real["function"].unique())
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}

df_real["label_id"] = df_real["function"].map(lab2id)


# ================================
# 4. TRAIN/TEST SPLIT (SAFE)
# ================================
train_df, test_df = train_test_split(
    df_real,
    test_size=0.20,
    random_state=42,
    stratify=df_real["label_id"],  # SAFE because all classes ≥ 2
)

print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))


# ================================
# Dataset class
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


train_ds = LookDataset(train_df["sample"], train_df["label_id"])
test_ds  = LookDataset(test_df["sample"],  test_df["label_id"])


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
# 6. TRAINING ARGUMENTS
# ================================
args = TrainingArguments(
    output_dir="original_only_results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",     # Transformers v5
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


# ================================
# 7. TRAIN
# ================================
trainer.train()


# ================================
# 8. EVALUATE ON REAL-ONLY
# ================================
pred = trainer.predict(test_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

print("\n=== TEST RESULTS (REAL ONLY) ===")
print("Accuracy:", round(acc, 3))
print("Macro F1:", round(macro_f1, 3))
print()

rep = classification_report(
    [id2lab[i] for i in y_true],
    [id2lab[i] for i in y_pred],
    digits=3,
    zero_division=0,
)
print(rep)

