from dataclasses import dataclass

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_accuracy, multiclass_precision, multiclass_recall
from torchmetrics.functional.classification import multilabel_f1_score, multilabel_accuracy, multilabel_precision, multilabel_recall
from .generic import BaseLearnerConfig, BaseLearner


@dataclass
class DownstreamClassificationLearnerConfig(BaseLearnerConfig):
    finetune_layers: int = 1
    n_classes: int = None
    multilabel: bool = False
    encoder_weights: str = None


class DownstreamClassificationLearner(BaseLearner):
    def __init__(self, model, config: DownstreamClassificationLearnerConfig):
        super().__init__()
        self.model = model
        self.config = config

        self.clf_head = nn.Linear(self.model.get_hidden_size(), config.n_classes)

    def forward(self, batch):
        """`batch` is a list of samples from DomainDataset"""
        dialogues = [dia for dia, _ in batch]
        targets = torch.stack([tar for _, tar in batch], dim=0)
        
        embeddings = self.model(dialogues)  # (B, H)
        logits = self.clf_head(embeddings)  # (B, n_classes)
        
        if self.config.multilabel:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        else:
            loss = F.cross_entropy(logits, targets, reduction='mean')

        if self.config.multilabel:
            metrics = {
                'f1_score': multilabel_f1_score(logits, targets, average='macro', num_labels=self.config.n_classes),
                'accuracy': multilabel_accuracy(logits, targets, average='macro', num_labels=self.config.n_classes),
                'precision': multilabel_precision(logits, targets, average='macro', num_labels=self.config.n_classes),
                'recall': multilabel_recall(logits, targets, average='macro', num_labels=self.config.n_classes),
            }
        else:
            targets = torch.argmax(targets, dim=1)
            metrics = {
                'f1_score': multiclass_f1_score(logits, targets, average='macro', num_classes=self.config.n_classes),
                'accuracy': multiclass_accuracy(logits, targets, average='macro', num_classes=self.config.n_classes),
                'precision': multiclass_precision(logits, targets, average='macro', num_classes=self.config.n_classes),
                'recall': multiclass_recall(logits, targets, average='macro', num_classes=self.config.n_classes),
            }

        return loss, metrics

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
    
    def validation_step(self, batch, batch_idx):
        loss, metrics = self.forward(batch)
        
        metrics['loss'] = loss
        metrics_prefix = {}
        for k, v in metrics.items():
            metrics_prefix[f'val/{k}'] = v

        self.log_dict(
            dictionary=metrics_prefix,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size,
            sync_dist=True
        )
