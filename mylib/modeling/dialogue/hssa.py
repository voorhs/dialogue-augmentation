from dataclasses import dataclass
from torch import nn
from ...utils.modeling import AveragePooling, SelfAttentionPooling
from ..hssa import HSSAModel, HSSAConfig, HSSATokenizer
from .base_dialogue_model import BaseDialogueModel


@dataclass
class HSSADialogueEncoderConfig:
    hf_model: str = 'sentence-transformers/all-mpnet-base-v2'
    pooling: str = 'avg'    # 'avg' or `att`


class HSSADialogueEncoder(nn.Module, BaseDialogueModel):
    def __init__(
            self,
            config: HSSADialogueEncoderConfig,
        ):
        super().__init__()

        self.config = config
        self.hf_config = HSSAConfig()

        self.model = HSSAModel.from_pretrained(config.hf_model, config=self.hf_config)
        self.tokenizer = HSSATokenizer.from_pretrained(config.hf_model)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if config.pooling == 'avg':
            self.pool = AveragePooling()
        elif config.pooling == 'att':
            self.pool = SelfAttentionPooling(self.get_hidden_size())
        else:
            raise ValueError(f'unknown pooling: {config.pooling}')
    @property
    def device(self):
        return self.model.device

    def forward(self, batch):
        """
        returned shape: (B, H)
        """

        tokenized = self.tokenizer(batch).to(self.device)
        hidden_states = self.model(**tokenized)     # (B, T, S, H)
        hidden_states = hidden_states[:, :, 0, :]   # (B, T, H), cls tokens of utterances
        outputs = self.pool(
            last_hidden_state=hidden_states,
            attention_mask=tokenized['utterance_mask']
        )
        
        return outputs
    
    def get_hidden_size(self):
        return self.hf_config.hidden_size

    @staticmethod
    def _tokenize(tokenizer, batch):
        return tokenizer(batch)
