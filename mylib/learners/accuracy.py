import torch
import numpy as np


def all_accuracies(scores):
    res = {}
    for k in [1, 3, 5, 10, 20]:
        if scores.shape[0] <= k:
            continue
        topk_indicators = [i in top for i, top in enumerate(torch.topk(scores, k=k, dim=1).indices)]
        res[f'accuracy@{k}'] = np.mean(topk_indicators)
    return res
