from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .generic import BaseLearnerConfig, BaseLearner
from .loss import contrastive_loss
from .accuracy import all_accuracies
from ..embedding_benchmarks import all_embedding_metrics


@dataclass
class DialogueEncoderLearnerConfig(BaseLearnerConfig):
    temperature: float = 0.1
    loss: str = 'contrastive_symmetric'   # 'contrastive_cross', 'contrastive_symmetric', 'contrastive_bce'
    finetune_layers: int = 1
    multilabel: bool = False


class DialogueEncoderLearner(BaseLearner):
    def __init__(self, model, config: DialogueEncoderLearnerConfig):
        super().__init__()
        self.model = model
        self.config = config

        # list of (embedding, target) pairs for validation
        self.multiwoz_train = []
        self.multiwoz_validation = []
        self.bitod_train = []
        self.bitod_validation = []
        self.sgd_train = []
        self.sgd_validation = []

    def forward(self, batch):
        return self._contrastive_step(batch)

    def _contrastive_step(self, batch):
        origs, positives = batch

        # encode all dialogues
        origs_enc = F.normalize(self.model(origs), dim=1)                   # (B, H)
        positives_enc = F.normalize(self.model(positives), dim=1)           # (B, H)

        # randomly swap correponding x and y to prevent from learning grammatics
        batch_size = len(origs)
        swap_or_not = torch.randn(batch_size) > 0
        origs_enc[swap_or_not], positives_enc[swap_or_not] = positives_enc[swap_or_not], origs_enc[swap_or_not]

        # contrastive loss
        pairwise_scores = (origs_enc @ positives_enc.T) / self.config.temperature
        loss = contrastive_loss(pairwise_scores, self.config.loss)
        
        # compute metric: retrieval accuracy
        metrics = all_accuracies(pairwise_scores)

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
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        dialogues, targets = batch

        targets = torch.stack(targets, dim=0).detach().cpu()
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
        
        multiwoz_metrics = all_embedding_metrics(self.multiwoz_train, self.multiwoz_validation, self.config.multilabel)
        bitod_metrics = all_embedding_metrics(self.bitod_train, self.bitod_validation, self.config.multilabel)
        sgd_metrics = all_embedding_metrics(self.sgd_train, self.sgd_validation, self.config.multilabel)

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
