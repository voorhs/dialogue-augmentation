import torch.nn as nn

from ...utils.training import LightningCkptLoadable
from ...utils.modeling import Projector
from .context_encoders import ContextEncoderConcat
from .target_encoders import TargetEncoder
from .config import PairwiseModelConfig


class Pairwise(nn.Module, LightningCkptLoadable):
    def __init__(self, model, config: PairwiseModelConfig):
        super().__init__()

        self.config = config

        if config.symmetric:
            target_encoder = ContextEncoderConcat(model, config, for_target=True)
        else:
            target_encoder = TargetEncoder(model, config)
        
        self.target_encoder = target_encoder
        self.context_encoder = ContextEncoderConcat(model, config)
        
        self.context_projector = Projector(
            input_size=self.context_encoder.get_encoding_size(),
            output_size=self.config.projection_size,
            dropout=config.projector_dropout
        )
        self.target_projector = Projector(
            input_size=self.target_encoder.get_encoding_size(),
            output_size=self.config.projection_size,
            dropout=config.projector_dropout
        )

    @property
    def device(self):
        return self.target_encoder.device
    
    def forward(self, batch):
        context_batch = []
        target_batch = []
        for pair in batch:
            context_batch.append(pair['context'])
            target_batch.append(pair['target'])
        
        target_encodings = self.target_encoder(target_batch)
        context_encodings = self.context_encoder(context_batch)

        context_encodings = self.context_projector(context_encodings)
        target_encodings = self.target_projector(target_encodings)

        return context_encodings, target_encodings

    # @torch.no_grad()
    # def score(self, dialogue, temperature=1):
    #     batch = self.make_batch_from_dia(dialogue)
    #     logits = self.get_logits(batch, temperature)
    #     return F.softmax(logits, dim=1).diag().log10().mean().cpu().item()
    
    @staticmethod
    def make_batch_from_dia(dialogue):
        batch = []
        for i in range(1, len(dialogue)):
            batch.append({
                'context': dialogue[:i],
                'target': dialogue[i]
            })
        return batch
