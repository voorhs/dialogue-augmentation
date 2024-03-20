from typing import List

from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch

from .poolers import AveragePooling


class mySentenceTransformer(nn.Module):
    """Imitation of SentenceTransformers (https://www.sbert.net/)"""

    def __init__(
            self,
            hf_model='sentence-transformers/all-mpnet-base-v2',
        ):
        """If `pooling=False`, then instead of sentence embeddings forward will return list of token embeddings."""
        super().__init__()
        self.hf_model = hf_model

        self.model = AutoModel.from_pretrained(hf_model)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

        self.pooler = AveragePooling()

    def forward(self, sentences: List[str]):
        inputs = self.tokenizer(
            sentences,
            padding='longest',
            return_tensors='pt',
            truncation=True
        ).to(self.model.device)
        output = self.model(**inputs)
        return self.pooler(output.last_hidden_state, inputs['attention_mask'])

    def get_sentence_embedding_size(self):
        return self.model.config.hidden_size
    
    @property
    def device(self):
        return self.model.device
