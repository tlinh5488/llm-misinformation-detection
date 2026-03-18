import pandas as pd
import os
import torch
import numpy as np
import random

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ======================
# SEED
# ======================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def main():

    seed_everything(42)

    print("CUDA:", torch.cuda.is_available())

    # ======================
    # PATH
    # ======================
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
    model_dir = os.path.join(BASE_DIR, "results", "roberta_clean")

    os.makedirs(model_dir, exist_ok=True)

    # ======================
    # LOAD DATA
    # ======================
    df = pd.read_csv(dataset_path, encoding="latin1")
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]

    n = min(len(real), len(fake))
    df = pd.concat([
        real.sample(n, random_state=42),
        fake.sample(n, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # ======================
    # SPLIT
    # ======================
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # ======================
    # TOKENIZER (CHUẨN)
    # ======================
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=96
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # ======================
    # MODEL (QUAN TRỌNG)
    # ======================
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

    # ======================
    # METRICS
    # ======================
    def compute_metrics(pred):
        logits = torch.tensor(pred.predictions)
        probs = torch.softmax(logits, dim=1)[:, 1]

        preds = (probs > 0.5).int().numpy()
        labels = pred.label_ids

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )

        acc = accuracy_score(labels, preds)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # ======================
    # TRAINING
    # ======================
    training_args = TrainingArguments(

        output_dir=model_dir,

        # GPU MX450 (2GB)
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,

        fp16=torch.cuda.is_available(),

        num_train_epochs=3,

        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",

        logging_steps=100,
        report_to="none"
    )

    # ======================
    # TRAINER
    # ======================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # ======================
    # TRAIN
    # ======================
    trainer.train()

    # ======================
    # EVALUATE
    # ======================
    results = trainer.evaluate()
    print("\nFINAL RESULT:", results)

    # ======================
    # SAVE (CHUẨN)
    # ======================
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("Saved to:", model_dir)


if __name__ == "__main__":
    main()