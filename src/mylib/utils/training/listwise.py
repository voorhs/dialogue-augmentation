from torch.utils.data import Dataset
from typing import Any, Literal, Tuple
import math
import json
from dataclasses import dataclass
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from .generic import BaseLearner, BaseLearnerConfig


class DialogueDataset(Dataset):
    chunk_size = 512
    def __init__(self, path, split: Literal['train', 'test', 'val'], fraction=1.):
        self.split = split
        self.path = path

        if split == 'train':
            max_n_chunks = 880
        elif split == 'test' or split == 'val':
            max_n_chunks = 48

        if isinstance(fraction, float):
            self.fraction = min(1., max(0., fraction))
            self.n_chunks = math.ceil(self.fraction * max_n_chunks)
        elif isinstance(fraction, int):
            self.fraction = min(max_n_chunks, max(1, fraction))
            self.n_chunks = fraction
        else:
            raise ValueError('fraction must be int or float')

        self.len = self.n_chunks * self.chunk_size
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one dialogue, represented with an object of the following schema:
        ```
        {
            "type": "array",
            "items":
            {
                "type": "object",
                "properties":
                {
                    "utterance": {"type": "string"},
                    "speaker": {"type": "number"}
                }
            }
        }
        ```"""
        i_chunk = math.floor(i / self.chunk_size)
        idx_within_chunk = i % self.chunk_size
        item = json.load(open(f'{self.path}/dialogues/{self.split}/{i_chunk}.json', 'r'))[idx_within_chunk]
        return item


@dataclass
class ListwiseLearnerConfig(BaseLearnerConfig):
    finetune_layers: int = 3


class ListwiseLearner(BaseLearner):
    def __init__(self, model, config: ListwiseLearnerConfig):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        loss, metric = self.forward(batch)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        self.log(
            name='train_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, metric = self.forward(batch)
        self.log(
            name='val_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        self.log(
            name='val_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
    
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
