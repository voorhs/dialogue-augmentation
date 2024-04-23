from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ...utils.modeling import CLSPooling, AveragePooling, SelfAttentionPooling, LastTokenPooling
from .base_dialogue_model import BaseDialogueModel


@dataclass
class BaselineDialogueEncoderConfig:
    hf_model: str = 'google-bert/bert-base-uncased'
    pooling: str = 'cls' # 'avg' or 'cls' or 'att' or 'last'
    truncation: bool = False
    max_length: int = 512


class BaselineDialogueEncoder(nn.Module, BaseDialogueModel):
    def __init__(self, config: BaselineDialogueEncoderConfig, **automodel_kwargs):
        super().__init__()

        self.config = config

        self.model = AutoModel.from_pretrained(config.hf_model, **automodel_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.hf_model)

        if self.config.pooling == 'cls':
            self.pooler = CLSPooling()
        elif self.config.pooling == 'avg':
            self.pooler = AveragePooling()
        elif self.config.pooling == 'att':
            self.pooler = SelfAttentionPooling(self.model.config.hidden_size)
        elif self.config.pooling == 'last':
            self.pooler = LastTokenPooling()
        else:
            raise ValueError('unknown pooler name')

    @property
    def device(self):
        return self.model.device

    @staticmethod
    def _parse(dia):
        """
        return list of strings of the following format: 
        [
            "0 Hello, dear!",
            "1 Hi.",
            "0 How can I help you?",
            "1 Tell me a joke."
        ]
        """
        return [f'{item["speaker"]} {item["utterance"]}' for item in dia]

    @staticmethod
    def _tokenize(tokenizer: AutoTokenizer, batch, truncation=False, max_length=512):
        """
        1. convert each dialogue to a single string of the following format: 
            "0 Hello, dear![SEP]1 Hi.[SEP]0 How can I help you?[SEP]1 Tell me a joke."
        2. tokenize every string with the standard tokenizer of the provided hf model
        """
        sep = tokenizer.sep_token
        if sep is None:
            sep = '\n'
        parsed = [sep.join(BaselineDialogueEncoder._parse(dia)) for dia in batch]
        inputs = tokenizer(parsed, padding=True, max_length=max_length, return_tensors='pt', truncation=truncation)
        return inputs
    
    def forward(self, batch):
        """
        input: batch of dialogues
        output: (B,H)
        """
        inputs = self._tokenize(
            self.tokenizer,
            batch,
            truncation=self.config.truncation,
            max_length=self.config.max_length
        ).to(self.device)
        outputs = self.model(**inputs)
        encodings = self.pooler(outputs.last_hidden_state, inputs.attention_mask)
        return encodings
    
    def get_all_hidden_states(self, batch, pooler, n_last):
        """
        input: batch of dialogues
        output: embeddings (B,H), hidden_states (n_last,B,H), hidden_embeddings (n_last,B,H)
        """
        inputs = self._tokenize(
            self.tokenizer,
            batch,
            truncation=self.config.truncation,
            max_length=self.config.max_length
        ).to(self.device)
        outputs = self.model(**inputs)
        embeddings = self.pooler(outputs.last_hidden_state, inputs.attention_mask)

        hidden_states = []
        for hidden_state in outputs.hidden_states[-n_last:]:
            hidden_states.append(pooler(hidden_state, inputs.attention_mask))
        hidden_states = torch.stack(hidden_states, dim=0)

        if not isinstance(self.pooler, AveragePooling):
            hidden_embeddings = []
            for hidden_state in outputs.hidden_states[-n_last:]:
                hidden_embeddings.append(self.pooler(hidden_state, inputs.attention_mask))
            hidden_embeddings = torch.stack(hidden_embeddings, dim=0)
        else:
            hidden_embeddings = None

        return embeddings, hidden_states, hidden_embeddings

    def get_hidden_size(self):
        return self.model.config.hidden_size
