import torch
import os
from transformers import BertTokenizerFast, BertForSequenceClassification


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "results", "bert_results")


tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=32
    )

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        pred = torch.argmax(logits, dim=1).item()

    if pred == 1:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"


if __name__ == "__main__":

    text = input("Enter news text: ")

    result = predict(text)

    print("Prediction:", result)