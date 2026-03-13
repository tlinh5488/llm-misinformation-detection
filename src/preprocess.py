import os
import pandas as pd
import re

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

real_path = os.path.join(BASE_DIR, "data", "raw", "real_news.csv")
fake_path = os.path.join(BASE_DIR, "data", "raw", "fake_news.csv")

real = pd.read_csv(real_path)
fake = pd.read_csv(fake_path)

real["label"] = 0
fake["label"] = 1

dataset = pd.concat([real, fake])

def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text

dataset["text"] = dataset["text"].apply(clean_text)

output_path = os.path.join(BASE_DIR, "data", "processed", "dataset_clean.csv")

dataset.to_csv(output_path, index=False)

print("Preprocessing finished")