import torch
import torch.nn.functional as F


def prepare_data(train_dataset, val_dataset):
    X_train = torch.stack([emb for emb, _ in train_dataset], dim=0)
    Y_train = torch.stack([tar for _, tar in train_dataset], dim=0)
    X_val = torch.stack([emb for emb, _ in val_dataset], dim=0)
    Y_val = torch.stack([tar for _, tar in val_dataset], dim=0)
    
    X_train = F.normalize(X_train, dim=1)
    X_val = F.normalize(X_val, dim=1)

    return X_train, Y_train, X_val, Y_val
