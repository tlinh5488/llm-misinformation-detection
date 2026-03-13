import pandas as pd
import os

from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def main():

    # ======================
    # PATH
    # ======================

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    dataset_path = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")

    model_dir = os.path.join(BASE_DIR, "results", "roberta_results")

    os.makedirs(model_dir, exist_ok=True)


    # ======================
    # LOAD DATASET
    # ======================

    df = pd.read_csv(dataset_path, encoding="latin1")

    print("Original dataset:", len(df))
    print(df["label"].value_counts())


    # ======================
    # BALANCE DATASET
    # ======================

    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]

    sample_size = min(len(real), len(fake))

    real_sample = real.sample(n=sample_size, random_state=42)
    fake_sample = fake.sample(n=sample_size, random_state=42)

    df = pd.concat([real_sample, fake_sample])

    df = df.sample(frac=1, random_state=42)

    print("Balanced dataset:", len(df))


    # ======================
    # SPLIT
    # ======================

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))


    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)


    # ======================
    # TOKENIZER
    # ======================

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=32
        )


    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)


    # ======================
    # DYNAMIC PADDING
    # ======================

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )


    # ======================
    # MODEL
    # ======================

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )


    # ======================
    # METRICS
    # ======================

    def compute_metrics(pred):

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="binary"
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

        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,

        num_train_epochs=1,

        logging_steps=200,

        save_strategy="no",

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

    print(results)


    # ======================
    # SAVE MODEL
    # ======================

    trainer.save_model(model_dir)

    tokenizer.save_pretrained(model_dir)

    print("RoBERTa model saved to:", model_dir)


if __name__ == "__main__":
    main()