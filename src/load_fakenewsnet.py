import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_dir = os.path.join(BASE_DIR, "data", "raw")

files = [
    "gossipcop_fake.csv",
    "gossipcop_real.csv",
    "politifact_fake.csv",
    "politifact_real.csv"
]

data = []

for file in files:

    path = os.path.join(data_dir, file)

    df = pd.read_csv(path)

    if "fake" in file:
        label = 1
    else:
        label = 0

    for text in df["title"]:
        data.append([text, label])

dataset = pd.DataFrame(data, columns=["text","label"])

output = os.path.join(BASE_DIR, "data", "processed", "fakenewsnet_dataset.csv")

dataset.to_csv(output, index=False)

print("FakeNewsNet dataset created")