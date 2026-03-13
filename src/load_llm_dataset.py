import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

path = os.path.join(
    BASE_DIR,
    "data/raw/synthetic-gpt-3.5-turbo_politifact_paraphrase_generation_processed.csv"
)

df = pd.read_csv(path)

llm = pd.DataFrame({
    "text": df["synthetic_misinformation"],
    "label": 1
})

output = os.path.join(BASE_DIR, "data/processed/llm_dataset.csv")

llm.to_csv(output, index=False)

print("LLM dataset created")