from torch import nn

from ...utils.modeling import mySentenceTransformer
from .config import PairwiseModelConfig


class BaseTargetEncoder(nn.Module):
    def get_encoding_size(self):
        raise NotImplementedError()

    @property
    def device(self):
        raise NotImplementedError()
    

class TargetEncoder(BaseTargetEncoder):
    def __init__(self, sentence_encoder: mySentenceTransformer, config: PairwiseModelConfig):
        super().__init__()

        self.config = config
        self.sentence_encoder = sentence_encoder
    
    def forward(self, batch):
        return self.sentence_encoder([item['utterance'] for item in batch])

    @property
    def device(self):
        return self.sentence_encoder.device
    
    def get_encoding_size(self):
        return self.sentence_encoder.get_sentence_embedding_size()
