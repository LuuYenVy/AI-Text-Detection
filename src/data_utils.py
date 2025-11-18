import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])

def load_and_split_data(train_path, test_path, text_cols, label_col, val_size=0.2, random_state=42):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_texts_full = (train_df[text_cols[0]] + " " + train_df[text_cols[1]]).tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts_full,
        train_df[label_col].tolist(),
        test_size=val_size,
        random_state=random_state
    )
    return train_texts, val_texts, train_labels, val_labels, test_df
