from dataclasses import dataclass

from torch import nn
from transformers import AutoModel, AutoTokenizer

from ...utils.modeling import CLSPooling, AveragePooling, SelfAttentionPooling
from .base_dialogue_model import BaseDialogueModel


@dataclass
class BaselineDialogueEncoderConfig:
    hf_model: str = 'google-bert/bert-base-uncased'
    pooling: str = 'cls' # 'avg' or 'cls' or 'att'


class BaselineDialogueEncoder(nn.Module, BaseDialogueModel):
    def __init__(self, config: BaselineDialogueEncoderConfig):
        super().__init__()

        self.config = config

        self.model = AutoModel.from_pretrained(config.hf_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.hf_model)

        if self.config.pooling == 'cls':
            self.pooler = CLSPooling()
        elif self.config.pooling == 'avg':
            self.pooler = AveragePooling()
        elif self.config.pooling == 'att':
            self.pooler = SelfAttentionPooling(self.model.config.hidden_size)
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
    def _tokenize(tokenizer, batch):
        """
        1. convert each dialogue to a single string of the following format: 
            "0 Hello, dear![SEP]1 Hi.[SEP]0 How can I help you?[SEP]1 Tell me a joke."
        2. tokenize every string with the standard tokenizer of the provided hf model
        """
        sep = tokenizer.sep_token
        parsed = [sep.join(BaselineDialogueEncoder._parse(dia)) for dia in batch]
        inputs = tokenizer(parsed, padding='longest', return_tensors='pt')
        return inputs
    
    def forward(self, batch):
        """
        input: batch of dialogues
        output: (B,H)
        """
        inputs = self._tokenize(self.tokenizer, batch).to(self.device)
        outputs = self.model(**inputs)
        encodings = self.pooler(outputs.last_hidden_state, inputs.attention_mask)
        return encodings

    def get_hidden_size(self):
        return self.model.config.hidden_size
