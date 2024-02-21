import random

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .prune import _load_pairwise_cat, _cluster, Pairwise, get_similarities


class Shuffler:
    def __init__(
            self,
            ckpt_path='./logs/comet/pairwise-model/7a5dd2169d3b49d696a67ba06af43f0e/checkpoints/last.ckpt',
            device='cpu',
            thresh=-np.inf
        ):
        self.thresh = thresh
        self.model = _load_pairwise_cat(ckpt_path, device)
    
    def __call__(self, dialogues):
        res = []
        for dia in tqdm(dialogues, desc='shuffling dialogues'):
            aug, score = self._shuffle(self.model, dia)
            res.append(aug if score >= self.thresh else None)
        return res

    @staticmethod
    @torch.no_grad()
    def _shuffle(model: Pairwise, dia):
        if len(dia) < 12:
            return None, -np.inf

        end = len(dia) // 3
        start = 4

        variations = []
        for n_clusters in range(start, end+1):
            clusterwise_uts = _cluster(model, dia, n_clusters)
            for i_try in range(n_clusters):
                random.shuffle(clusterwise_uts)
                aug = []
                for ut_ids in clusterwise_uts:
                    aug.extend([dia[i] for i in ut_ids])
                score = score(model, aug)
                variations.append((aug, score))
        
        return max(variations, key=lambda x: x[1])
        


@torch.no_grad()
def score(model, dia):
    logits = get_similarities(model, dia)
    return F.softmax(logits, dim=1).diag().log10().mean().cpu().item()
