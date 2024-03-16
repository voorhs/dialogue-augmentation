from dataclasses import dataclass


@dataclass
class PairwiseModelConfig:
    projection_size: int = 128
    symmetric: bool = False
    context_size: int = 3
    n_speakers: int = 2
    speaker_embedding_dim = 8
    projector_dropout: float = 0.1
    hf_model: str = 'aws-ai/dse-bert-base'


@dataclass
class SimplifiedPairwiseModelConfig:
    context_size: int = 2
    hf_model: str = 'aws-ai/dse-bert-base'
