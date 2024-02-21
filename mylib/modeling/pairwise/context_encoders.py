import torch
from torch import nn
import torch.nn.functional as F

from .config import PairwiseModelConfig
from ...utils.modeling.generic import mySentenceTransformer


class BaseContextEncoder(nn.Module):    
    def get_encoding_size(self):
        raise NotImplementedError()


class ContextEncoderConcat(BaseContextEncoder):
    def __init__(self, sentence_encoder: mySentenceTransformer, config: PairwiseModelConfig, for_target=False):
        super().__init__()

        self.for_target = for_target
        self.config = config
        self.sentence_encoder = sentence_encoder
    
    def get_embeddings(self, batch):
        uts = []
        lens = []
        for dia in batch:
            i = 0 if self.for_target else -1
            cur_uts = [f"{dia[i]['speaker']} {item['utterance']}" for item in dia]
            uts.extend(cur_uts)
            lens.append(len(cur_uts))
        
        sentence_embeddings = self.sentence_encoder(uts)
        sentence_embeddings = list(torch.unbind(sentence_embeddings, dim=0))

        return sentence_embeddings, lens

    def forward(self, batch):
        sentence_embeddings, lens = self.get_embeddings(batch)

        res = []
        for i in range(len(batch)):
            start = sum(lens[:i])
            end = start + lens[i]
            enc = self.pad_missing_utterances(sentence_embeddings[start:end], lens[i])
            res.append(enc)

        return torch.stack(res, dim=0)
    
    def pad_missing_utterances(self, sentences, n_actual_utterances):
        """
        if actual given context is smaller than defined max context length,
        then zero out embedding entries, that correspond to missing uttarances"""
        
        d = self.sentence_encoder.get_sentence_embedding_size()
        n_zeros_to_pad = (self.config.context_size - n_actual_utterances) * d
        
        flattened = torch.cat(sentences)
        return F.pad(flattened, pad=(0, n_zeros_to_pad), value=0)

    def get_encoding_size(self):
        return self.sentence_encoder.get_sentence_embedding_size() * self.config.context_size


# deprecated and not supported
class ContextEncoderEMA(BaseContextEncoder):
    def __init__(self, sentence_encoder: mySentenceTransformer, context_size, tau):
        super().__init__()

        self.sentence_encoder = sentence_encoder
        self.context_size = context_size
        self.tau = tau

    def forward(self, batch):
        uts = []
        lens = []
        for dia in batch:
            cur_uts = [item['utterance'] for item in dia]
            uts.extend(cur_uts)
            lens.append(len(cur_uts))
        
        sentence_embeddings = self.sentence_encoder(uts)
        return self._ema(sentence_embeddings, lens, self.tau)
    
    @staticmethod
    def _ema(sentence_embeddings, lens, tau):
        res = []
        for i in range(len(lens)):
            start = sum(lens[:i])
            end = start + lens[i]
            embs = sentence_embeddings[start:end]
            
            last_ut = embs[-1]

            if lens[i] > 1:
                prev_uts = embs[-2]
            else:
                prev_uts = torch.zeros_like(last_ut)
            for prev_ut in embs[-3:-lens[i]-1:-1]:
                prev_uts = tau * prev_uts + (1 - tau) * prev_ut
            
            res.append(torch.cat([prev_uts, last_ut]))
        
        return res

    def get_encoding_size(self):
        return 2 * self.sentence_encoder.get_sentence_embedding_size()


# deprecated and not supported
class ContextEncoderDM(BaseContextEncoder):
    def __init__(self, dialogue_model, tau):
        super().__init__()
        self.tau = tau
        self.dialogue_model = dialogue_model
    
    def forward(self, batch):
        hidden_states = self.dialogue_model(batch)
        
        lens = [len(context) for context in batch]
        
        sentence_embeddings = []
        for hs, length in zip(hidden_states, lens):
            sentence_embeddings.extend(torch.unbind(hs, dim=0)[:length])
        encodings = ContextEncoderEMA._ema(sentence_embeddings, lens, self.tau)
        
        return encodings

    def get_encoding_size(self):
        return 2 * self.dialogue_model.model.config.hidden_size
