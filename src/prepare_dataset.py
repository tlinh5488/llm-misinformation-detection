import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

file_path = os.path.join(
    BASE_DIR,
    "data/raw/synthetic-gpt-3.5-turbo_politifact_paraphrase_generation_processed.csv"
)

df = pd.read_csv(file_path)

human_fake = df["news_text"]
llm_fake = df["synthetic_misinformation"]

human_df = pd.DataFrame({
    "text": human_fake,
    "label": 1
})

llm_df = pd.DataFrame({
    "text": llm_fake,
    "label": 1
})

dataset = pd.concat([human_df, llm_df])

output_path = os.path.join(BASE_DIR, "data/processed/final_dataset.csv")

dataset.to_csv(output_path, index=False)

print("Dataset created successfully")