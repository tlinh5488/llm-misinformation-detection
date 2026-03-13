import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

fakenewsnet = os.path.join(BASE_DIR, "data/processed/fakenewsnet_dataset.csv")
llm = os.path.join(BASE_DIR, "data/processed/llm_dataset.csv")

df1 = pd.read_csv(fakenewsnet)
df2 = pd.read_csv(llm)

merged = pd.concat([df1, df2])

output = os.path.join(BASE_DIR, "data/processed/final_dataset.csv")

merged.to_csv(output, index=False)

print("Final dataset created")