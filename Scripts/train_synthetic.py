import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ======================================================
# 1. LOAD SYNTHETIC DATA ONLY
# ======================================================
FILE = "/Users/aysasorahi/Documents/master/SLAM LAB/REZA/data/LOOK_data.ods"
df = pd.read_excel(FILE, engine="odf")

# Normalize labels
df["function"] = (
    df["function"]
    .astype(str)
    .str.replace(r"\s+", "", regex=True)
    .str.strip()
    .str.upper()
)

# Synthetic only (Label NOT NaN)
df = df[df["Label"].notna()].copy()

# Only minority functions
VALID = ["DIR", "DM", "INTJ"]
df = df[df["function"].isin(VALID)].reset_index(drop=True)

print("\nSYNTHETIC-only data distribution:")
print(df["function"].value_counts())

# ======================================================
# 2. LABEL MAPPING (3-CLASS)
# ======================================================
labels = sorted(VALID)
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}

df["label_id"] = df["function"].map(lab2id)

X = df["sample"].astype(str).values
y = df["label_id"].values

# ======================================================
# 3. CLASS WEIGHTS
# ======================================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y,
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

print("\nClass weights:", class_weights.tolist())

# ======================================================
# 4. DATASET
# ======================================================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class LookDataset(Dataset):
    def __init__(self, texts, labels):
        enc = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=128,
        )
        self.enc = {k: torch.tensor(v) for k, v in enc.items()}
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item

# ======================================================
# 5. WEIGHTED TRAINER
# ======================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_ = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fn(logits, labels_)
        return (loss, outputs) if return_outputs else loss

# ======================================================
# 6. STRATIFIED 2-FOLD CV (50 / 50)
# ======================================================
SEED = 42  # change manually for each run
set_seed(SEED)

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

    print("\n" + "=" * 50)
    print(f"FOLD {fold} / 2")
    print("=" * 50)

    train_ds = LookDataset(X[train_idx], y[train_idx])
    test_ds  = LookDataset(X[test_idx],  y[test_idx])

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels),
        id2label=id2lab,
        label2id=lab2id,
    )

    args = TrainingArguments(
        output_dir=f"results/synthetic_only/fold{fold}",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=labels,
            digits=2,
            zero_division=0,
        )
    )
