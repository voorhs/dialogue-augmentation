from dataclasses import dataclass

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multilabel_f1_score

import numpy as np

from .generic import BaseLearnerConfig, BaseLearner
from ..embedding_benchmarks import all_embedding_metrics


@dataclass
class DialogueEncoderLearnerConfig(BaseLearnerConfig):
    k: int = 1
    temperature: float = 0.1
    loss: str = 'contrastive_cross'   # 'contrastive_cross', 'contrastive_symmetric', 'contrastive_bce', 'multiwoz_service_clf'
    finetune_layers: int = 1
    dialogue_model: str = 'baseline'    # 'baseline' or 'hssa' (may be something else will emerge in future)


class DialogueEncoderLearner(BaseLearner):
    def __init__(self, model, config: DialogueEncoderLearnerConfig):
        super().__init__()
        self.model = model
        self.config = config

        # list of (embedding, target) pairs for multiwoz service clf (as validation)
        self.multiwoz_train = []
        self.multiwoz_validation = []

        if self.config.loss == 'multiwoz_service_clf':
            self.clf_head = nn.Linear(self.model.get_hidden_size(), 7)

    def forward(self, batch):
        return self._contrastive_step(batch)

    def _contrastive_step(self, batch):
        """`batch` is a list of samples from ContrastiveDataset"""
        origs = [sample['orig']['content'] for sample in batch]
        
        # select positives
        points = np.random.uniform(low=0, high=1, size=len(batch))
        counts = np.array([len(sample['pos']) for sample in batch])
        pos_indices = np.floor(points * counts).astype(np.int_)
        
        positives = [sample['pos'][i]['content'] for i, sample in zip(pos_indices, batch)]

        # encode all dialogues
        origs_enc = F.normalize(self.model(origs), dim=1)                   # (B, H)
        positives_enc = F.normalize(self.model(positives), dim=1)           # (B, H)

        # randomly swap correponding x and y to prevent from learning grammatics
        batch_size = len(batch)
        swap_or_not = torch.randn(batch_size) > 0
        origs_enc[swap_or_not], positives_enc[swap_or_not] = positives_enc[swap_or_not], origs_enc[swap_or_not]

        # contrastive loss
        pairwise_scores = (origs_enc @ positives_enc.T) / self.config.temperature
        loss = contrastive_loss(pairwise_scores, self.config.loss)
        
        # compute metric: retrieval accuracy
        metrics = all_accuracies(pairwise_scores)

        return loss, metrics

    def _multiwoz_service_clf_step(self, batch):
        """`batch` is a list of samples from MultiWOZServiceClfDataset"""
        dialogues = [dia for dia, _ in batch]
        targets = torch.stack([tar for _, tar in batch], dim=0)
        
        embeddings = self.model(dialogues)  # (B, H)
        logits = self.clf_head(embeddings)  # (B, 7)
        
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        metric = multilabel_f1_score(logits, targets, average='macro')

        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metrics = self.forward(batch)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size,
            sync_dist=True
        )
        self.log_dict(
            dictionary=metrics,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size,
            sync_dist=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        dialogues = [dia for dia, _ in batch]
        targets = torch.stack([tar for _, tar in batch], dim=0).detach().cpu()
        embeddings = self.model(dialogues).detach().cpu()
        res = list(zip(embeddings, targets))

        if dataloader_idx == 0:
            self.multiwoz_train.extend(res)
        elif dataloader_idx == 1:
            self.multiwoz_validation.extend(res)
    
    def on_validation_epoch_end(self) -> None:
        
        metrics = all_embedding_metrics(self.multiwoz_train, self.multiwoz_validation)

        for key, val in metrics.items():
            if not isinstance(val, torch.Tensor):
                metrics[key] = torch.tensor(val, device=self.device)

        self.log_dict(
            dictionary=metrics,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        self.multiwoz_train.clear()
        self.multiwoz_validation.clear()

    def configure_optimizers(self):
        optim_groups = self.get_parameter_groups()
        optimizer = AdamW(optim_groups, lr=self.config.lr, betas=self.config.betas)
        def lr_foo(step):
            warmup_steps = self.config.warmup_period
            periodic = self.config.do_periodic_warmup
            
            if warmup_steps is None:
                return 1
            if periodic:
                return (step % warmup_steps + 1) / warmup_steps
            else:
                return (step + 1) / warmup_steps if step < warmup_steps else 1

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}


def contrastive_loss(scores, name):
    if name == 'contrastive_symmetric':
        fn = contrastive_symmetric
    if name == 'contrastive_cross':
        fn = contrastive_cross
    if name == 'contrastive_bce':
        fn = contrastive_bce
    
    return fn(scores)


def contrastive_symmetric(scores):
    batch_size = scores.shape[0]
    targets = torch.eye(batch_size, device=scores.device)
    loss_1 = F.cross_entropy(scores, targets, reduction='mean')
    loss_2 = F.cross_entropy(scores.T, targets, reduction='mean')
    loss = loss_1 + loss_2
    return loss


def contrastive_cross(scores):
    scores = scores.exp()
    pos_scores = scores.diag()
    neg_scores1 = scores.sum(dim=0)
    neg_scores2 = scores.sum(dim=1)
    loss = (pos_scores / (neg_scores1 + neg_scores2 - pos_scores)).log().neg().sum()
    return loss


def contrastive_bce(scores):
    batch_size = scores.shape[0]
    targets = torch.eye(batch_size, device=scores.device)
    loss = F.binary_cross_entropy_with_logits(scores, targets, reduction='mean')
    return loss


def all_accuracies(scores):
    res = {}
    for k in [1, 3, 5, 10, 20]:
        if scores.shape[0] <= k:
            continue
        topk_indicators = [i in top for i, top in enumerate(torch.topk(scores, k=k, dim=1).indices)]
        res[f'train_accuracy@{k}'] = np.mean(topk_indicators)
    return res
