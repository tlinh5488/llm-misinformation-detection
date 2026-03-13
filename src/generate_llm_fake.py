import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

input_path = os.path.join(BASE_DIR, "data", "processed", "dataset_clean.csv")

df = pd.read_csv(input_path, encoding="latin1")

fake_news = df[df["label"] == 1]

generated = []

for text in fake_news["text"]:

    new_text = "Some reports claim that " + text

    generated.append(new_text)

llm_dataset = pd.DataFrame({
    "text": generated,
    "label": 1
})

output_path = os.path.join(BASE_DIR, "data", "llm_generated", "llm_fake_news.csv")

llm_dataset.to_csv(output_path, index=False)

print("LLM fake news generated")