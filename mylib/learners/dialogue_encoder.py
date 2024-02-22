from dataclasses import dataclass

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multilabel_f1_score

import numpy as np

from .generic import BaseLearnerConfig, BaseLearner
from .loss import contrastive_loss
from .accuracy import all_accuracies
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
        self.bitod_train = []
        self.bitod_validation = []
        self.sgd_train = []
        self.sgd_validation = []

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
        
        metrics['loss'] = loss
        metrics_prefix = {}
        for k, v in metrics.items():
            metrics_prefix[f'train/{k}'] = v

        self.log_dict(
            dictionary=metrics_prefix,
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

        elif dataloader_idx == 2:
            self.bitod_train.extend(res)
        elif dataloader_idx == 3:
            self.bitod_validation.extend(res)
            
        elif dataloader_idx == 4:
            self.sgd_train.extend(res)
        elif dataloader_idx == 5:
            self.sgd_validation.extend(res)
    
    def on_validation_epoch_end(self) -> None:
        
        multiwoz_metrics = all_embedding_metrics(self.multiwoz_train, self.multiwoz_validation)
        bitod_metrics = all_embedding_metrics(self.bitod_train, self.bitod_validation)
        sgd_metrics = all_embedding_metrics(self.sgd_train, self.sgd_validation)

        # https://github.com/Lightning-AI/pytorch-lightning/issues/18803
        res = {}
        for metrics, dataset_name in zip(
            [multiwoz_metrics, bitod_metrics, sgd_metrics],
            ['multiwoz', 'bitod', 'sgd']
        ):
            for key, val in metrics.items():
                if not isinstance(val, torch.Tensor):
                    res[f'{dataset_name}/{key}'] = torch.tensor(val, device=self.device)
                else:
                    res[f'{dataset_name}/{key}'] = val.to(self.device)

        self.log_dict(
            dictionary=res,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        self.multiwoz_train.clear()
        self.multiwoz_validation.clear()

        self.bitod_train.clear()
        self.bitod_validation.clear()

        self.sgd_train.clear()
        self.sgd_validation.clear()
