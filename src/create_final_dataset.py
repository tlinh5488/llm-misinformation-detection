import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

clean_path = os.path.join(BASE_DIR, "data", "processed", "dataset_clean.csv")
llm_path = os.path.join(BASE_DIR, "data", "llm_generated", "llm_fake_news.csv")

clean_df = pd.read_csv(clean_path, encoding="latin1")
llm_df = pd.read_csv(llm_path, encoding="latin1")

final_df = pd.concat([clean_df, llm_df])

output_path = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")

final_df.to_csv(output_path, index=False)

print("Final dataset created")