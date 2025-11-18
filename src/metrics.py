import torch
from sklearn.metrics import roc_auc_score, accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, probs > 0.5)
    return {"roc_auc": auc, "accuracy": acc}
