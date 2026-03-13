import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_dir = os.path.join(BASE_DIR, "FakeNewsNet", "dataset")

data = []

for root, dirs, files in os.walk(data_dir):

    for file in files:

        if file == "news content.json":

            path = os.path.join(root, file)

            with open(path, encoding="utf8") as f:

                article = json.load(f)

                text = article.get("text", "")

                if text.strip() == "":
                    continue

                if "fake" in root:
                    label = 1
                else:
                    label = 0

                data.append([text, label])

df = pd.DataFrame(data, columns=["text","label"])

output = os.path.join(BASE_DIR, "data", "processed", "fakenewsnet_dataset.csv")

df.to_csv(output, index=False)

print("FakeNewsNet dataset created")