from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def evaluate(predictions, labels):

    preds = predictions.argmax(-1)

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