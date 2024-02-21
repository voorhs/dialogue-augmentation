import torch
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

        self.speaker_embedding = nn.Embedding(
            config.n_speakers,
            config.speaker_embedding_dim
        )
    
    def forward(self, batch):
        uts = [item['utterance'] for item in batch]
        spe = [item['speaker'] for item in batch]
        
        sentence_embeddings = self.sentence_encoder(uts)
        speaker_ids = torch.tensor(spe, device=self.device)
        speaker_embeddings = self.speaker_embedding(speaker_ids)
        return torch.cat([sentence_embeddings, speaker_embeddings], dim=1)

    @property
    def device(self):
        return self.sentence_encoder.device
    
    def get_encoding_size(self):
        return self.config.speaker_embedding_dim + self.sentence_encoder.get_sentence_embedding_size()
