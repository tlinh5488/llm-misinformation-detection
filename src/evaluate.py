import os
import json
import pandas as pd
import torch

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    RobertaTokenizerFast,
    RobertaForSequenceClassification
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


# ======================
# PATH
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

BERT_MODEL_PATH = os.path.join(RESULTS_DIR, "bert_results")
ROBERTA_MODEL_PATH = os.path.join(RESULTS_DIR, "roberta_clean")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# METRICS
# ======================
def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


# ======================
# LOAD TEST SET (GIỐNG TRAIN ROBERTA)
# ======================
def prepare_test_set():
    df = pd.read_csv(DATASET_PATH, encoding="latin1")

    # giống train
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # 🔥 QUAN TRỌNG: balance dataset
    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]

    n = min(len(real), len(fake))

    df = pd.concat([
        real.sample(n, random_state=42),
        fake.sample(n, random_state=42)
    ])

    # shuffle giống train
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # split giống train
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    return test_df


# ======================
# EVALUATE
# ======================
def evaluate_model(model, tokenizer, test_df, batch_size=16):

    model.to(DEVICE)
    model.eval()

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    preds = []

    for i in range(0, len(texts), batch_size):

        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=96  # 🔥 giống train
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

        preds.extend(batch_preds)

    return compute_metrics(labels, preds)


# ======================
# MAIN
# ======================
def main():

    print("Preparing test set (reproduce train split)...")
    test_df = prepare_test_set()
    print("Test samples:", len(test_df))

    print("\nLoading models...")

    bert_tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_PATH)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)

    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL_PATH)
    roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH)

    print("\nEvaluating BERT...")
    bert_results = evaluate_model(bert_model, bert_tokenizer, test_df)
    print("BERT:", bert_results)

    print("\nEvaluating RoBERTa...")
    roberta_results = evaluate_model(roberta_model, roberta_tokenizer, test_df)
    print("RoBERTa:", roberta_results)

    # ======================
    # SAVE JSON
    # ======================
    final_results = {
        "BERT": bert_results,
        "RoBERTa": roberta_results
    }

    save_path = os.path.join(RESULTS_DIR, "final_evaluation.json")

    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n✅ Saved to:", save_path)


if __name__ == "__main__":
    main()