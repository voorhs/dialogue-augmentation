import torch.nn as nn

from ...utils.training import LightningCkptLoadable
from .config import SimplifiedPairwiseModelConfig


class SimplifiedPairwise(nn.Module, LightningCkptLoadable):
    def __init__(self, model, config: SimplifiedPairwiseModelConfig):
        super().__init__()

        self.config = config
        self.model = model

    @property
    def device(self):
        return self.model.device

    def forward(self, batch):
        context_batch = ['\n'.join([item['utterance'] for item in pair['context']]) for pair in batch]
        target_batch = [pair['target']['utterance'] for pair in batch]

        target_encodings = self.model(target_batch)
        context_encodings = self.model(context_batch)

        return context_encodings, target_encodings
