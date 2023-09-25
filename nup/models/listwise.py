import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from bisect import bisect_left
from .train_utils import LightningCkptLoadable, HParamsPuller
from typing import Literal


class RankerHead(nn.Module, HParamsPuller):
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()

        self.lin = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.ranker = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, x: torch.Tensor):
        x = F.gelu(self.lin(x)) + x
        x = self.dropout(x)
        x = self.ranker(x)
        return x.squeeze(-1)


class SortingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, ranks_logits, dia_lens):
        ranks_logprobs = F.log_softmax(ranks_logits, dim=1)
        _, T = ranks_logits.shape
        device = ranks_logits.device
        ranks_true = self._make_true_ranks(T, dia_lens, device)
        return self.loss_fn(ranks_logprobs, ranks_true)

    @staticmethod
    def _make_true_ranks(T, dia_lens, device):
        res = []
        
        for length in dia_lens:
            ranks = torch.linspace(1, 0, length, device=device)
            ranks = F.pad(ranks, pad=(0, T-length), value=0)
            ranks = ranks / ranks.sum()
            res.append(ranks)
        
        return torch.stack(res)


class SortingMetric:
    def __call__(self, ranks_logits, mask, dia_lens):
        unbinded_ranks_logits = self._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self._to_permutations(unbinded_ranks_logits)
        return 1-np.mean([self._normalized_inversions_count(perm) for perm in permutations])
       
    @staticmethod
    def _unbind_logits(logits, mask, dia_lens):
        """get list of tensors with logits corresponding to tokens(utterances) that are not padding ones only"""
        return logits[~mask].detach().cpu().split(dia_lens)

    @staticmethod
    def _to_permutations(unbinded_ranks_logits):
        """permutations with respect to descending order"""
        return [logits.argsort(descending=True) for logits in unbinded_ranks_logits]

    @staticmethod
    def _normalized_inversions_count(arr):
        """Function to count number of inversions in a permutation of 0, 1, ..., n-1."""
        n = len(arr)
        v = list(range(n))
        ans = 0
        for i in range(n):
            itr = bisect_left(v, arr[i])
            ans += itr
            del v[itr]
        max_inversions = n * (n - 1) / 2
        return ans / max_inversions


class ContrasterHead(nn.Module, HParamsPuller):
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        self.drop = nn.Dropout(dropout_prob)
        self.lin = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # x: (B, T, H)
        x = F.gelu(self.lin(self.drop(x))) + x
        x = torch.nn.functional.normalize(x, dim=2)
        return x


class PairingLoss(nn.Module, HParamsPuller):
    def __init__(self, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, hidden_states, mask, dia_lens):
        B, T, H = hidden_states.shape
        device = hidden_states.device
        
        context_indexes = torch.arange(T-1)
        target_indexes = torch.arange(1, T)
        pos_scores = torch.zeros_like(hidden_states)
        tmp = hidden_states[:, context_indexes, :] * hidden_states[:, target_indexes, :]
        pos_scores[:, context_indexes, :] += tmp
        pos_scores[:, target_indexes, :] += tmp
        
        mask2 = mask.clone()
        dia_lens_expanded = torch.tensor(dia_lens, device=device)
        mask2[torch.arange(B, device=device), dia_lens_expanded-1] = True
        pos_scores = pos_scores.sum(dim=2).masked_fill(mask2, -1e4).exp().masked_select(~mask2)

        hidden_states = hidden_states.view(-1, H)
        all_scores = hidden_states @ hidden_states.T
        
        mask3 = mask.view(1, -1) & torch.eye(B*T).bool().to(device)
        neg_scores = all_scores.masked_fill(mask3, -1e4).exp().sum(dim=1).masked_select(~mask2.view(-1))

        loss = pos_scores.div(neg_scores).log().neg()
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'mean':
            loss = loss.sum()
        return loss


class UtteranceSorter(nn.Module, LightningCkptLoadable, HParamsPuller):
    def __init__(self, dialogue_model, dropout_prob, with_contrastive=False):
        super().__init__()

        self.dialogue_model = dialogue_model
        self.dropout_prob = dropout_prob
        self.with_contrastive = with_contrastive
        
        self.ranker_head = RankerHead(dialogue_model.get_hidden_size(), dropout_prob)
        self.sorting_loss = SortingLoss()
        self.metric_fn = SortingMetric()
        
        if self.with_contrastive:
            self.contraster_head = ContrasterHead(dialogue_model.get_hidden_size(), dropout_prob)
            self.pairing_loss = PairingLoss()

    @property
    def device(self):
        return self.dialogue_model.device

    def get_logits(self, batch):
        hidden_states = self.dialogue_model(batch)
        return hidden_states, self.ranker_head(hidden_states)

    def forward(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        hidden_states, ranks_logits = self.get_logits(batch)

        # zero attention to padding token-utterances
        mask = self._make_mask(dia_lens, device)
        ranks_logits.masked_fill_(mask, -1e4)

        loss = self.sorting_loss(ranks_logits, dia_lens)
        if self.with_contrastive:
            hidden_states = self.contraster_head(hidden_states)
            pairing_loss = self.pairing_loss(hidden_states, mask, dia_lens)
            loss = (loss * 5 + pairing_loss) / 6
        metric = self.metric_fn(ranks_logits, mask, dia_lens)

        return loss, metric

    @torch.no_grad()
    def augment(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        ranks_logits = self.get_logits(batch)
        mask = self._make_mask(dia_lens, device)
        unbinded_ranks_logits = self.metric_fn._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self.metric_fn._to_permutations(unbinded_ranks_logits)

        return [[dia[i] for i in perm] for dia, perm in zip(batch, permutations)]

    @staticmethod
    def _make_mask(dia_lens, device):
        """this mask indicates padding tokens(utterances). used for ranking (not for transformer)"""
        T = max(dia_lens)
        dia_lens_expanded = torch.tensor(dia_lens, device=device)[:, None]
        max_dia_len_expanded = torch.arange(T, device=device)[None, :]
        return dia_lens_expanded <= max_dia_len_expanded
    

class ClfRankerHead(nn.Module, HParamsPuller):
    def __init__(self, hidden_size, n_classes, dropout_prob):
        super().__init__()

        self.lin = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.clf = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x: torch.Tensor):
        x = F.gelu(self.lin(x)) + x
        x = self.dropout(x)
        x = self.clf(x)
        return x


class ClfSortingLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, mask, dia_lens):
        """
        logits: (B, T, C)
        mask: (B, T), mask indicating padding utterances
        """
        device = logits.device

        for i, length in enumerate(dia_lens):
            logits[i, :, length:] = -1e4

        B, T, C = logits.shape
        logits = logits.view(-1, C)
        labels = self._make_true_labels(B, T, device)
        dirty_loss = self.loss_fn(logits, labels)
        mask = mask.view(-1)
        loss = dirty_loss[~mask].mean()
        return loss
    
    @staticmethod
    def _make_true_labels(B, T, device):
        return torch.arange(T, device=device)[None, :].expand(B, T).reshape(-1)


class ClfSortingMetric(SortingMetric):
    def __call__(self, probs_logits, mask, dia_lens):
        labs = torch.argmax(probs_logits, dim=-1)
        ranks = -labs

        return super().__call__(ranks, mask, dia_lens)


class ClfUtteranceSorter(nn.Module, LightningCkptLoadable, HParamsPuller):
    def __init__(self, dialogue_model, dropout_prob, max_n_uts):
        super().__init__()

        self.dialogue_model = dialogue_model
        self.dropout_prob = dropout_prob
        self.max_n_uts = max_n_uts
        
        self.clf_head = ClfRankerHead(dialogue_model.get_hidden_size(), max_n_uts, dropout_prob)
        self.sorting_loss = ClfSortingLoss()
        self.metric_fn = ClfSortingMetric()

    @property
    def device(self):
        return self.dialogue_model.device

    def get_logits(self, batch):
        hidden_states = self.dialogue_model(batch)
        return self.clf_head(hidden_states)

    def forward(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        # (B, T, C), where C is the # of max uts in dia
        probs_logits = self.get_logits(batch)
        mask = UtteranceSorter._make_mask(dia_lens, device)

        loss = self.sorting_loss(probs_logits, mask, dia_lens)
        metric = self.metric_fn(probs_logits, mask, dia_lens)

        return loss, metric

    @torch.no_grad()
    def augment(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        ranks_logits = self.get_logits(batch)
        mask = self._make_mask(dia_lens, device)
        unbinded_ranks_logits = self.metric_fn._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self.metric_fn._to_permutations(unbinded_ranks_logits)

        return [[dia[i] for i in perm] for dia, perm in zip(batch, permutations)]
    
