import torch
from torch import nn
from dataclasses import dataclass, asdict
from aux import myTransformerConfig, mySentenceTransformer, Projector, myTransformerBlock
from hssa import HSSAModel, HSSAConfig, HSSATokenizer
from transformers import MPNetModel, MPNetTokenizer
from transformers.models.mpnet.modeling_mpnet import create_position_ids_from_input_ids
from collections import defaultdict


@dataclass
class UtteranceTransformerDMConfig(myTransformerConfig):
    encoder_name: str = None
    max_dia_len: int = None
    embed_turn_ids: bool = True
    is_casual: bool = False


class UtteranceTransformerDM(nn.Module):
    def __init__(self, config: UtteranceTransformerDMConfig):
        super().__init__()

        self.encoder = mySentenceTransformer(model_name=config.encoder_name)
        self.max_dia_len = config.max_dia_len
        self.embed_turn_ids = config.embed_turn_ids

        sentence_embedding_dimension = self.encoder.get_sentence_embedding_size()
        projection_size = sentence_embedding_dimension // 2
        if self.embed_turn_ids:
            self.hidden_size = projection_size + 16
            self.turn_ids_embedding = nn.Embedding(config.max_dia_len, 8)
        else:
            self.hidden_size = projection_size + 8

        self.speaker_embeddings = nn.Embedding(2, 8)
        self.projector = Projector(sentence_embedding_dimension, projection_size)

        config.hidden_size = self.hidden_size
        config.intermediate_size = 4 * self.hidden_size
        self.transformer = nn.ModuleList([myTransformerBlock(config) for _ in range(config.n_layers)])

        self.config = config

    def get_hparams(self):
        return asdict(self.config)

    @property
    def device(self):
        return self.encoder.model.device

    def forward(self, batch):
        dia_lens = [len(dia) for dia in batch]

        inputs = []
        for dia in batch:
            speaker_ids = torch.tensor([item['speaker'] for item in dia], device=self.device, dtype=torch.long)
            speaker_embeddings = self.speaker_embeddings(speaker_ids)

            sentence_embeddings = self.encoder([item['utterance'] for item in dia])
            sentence_embeddings = self.projector(sentence_embeddings)

            utterance_embeddings = [sentence_embeddings, speaker_embeddings]
            
            if self.embed_turn_ids:
                turn_ids = torch.arange(len(dia), device=self.device)
                turn_ids_embeddings = self.turn_ids_embedding(turn_ids)
                utterance_embeddings += [turn_ids_embeddings]
            
            utterance_embeddings = torch.cat(utterance_embeddings, dim=1)
            utterance_embeddings = torch.unbind(utterance_embeddings)
            inputs.append(utterance_embeddings)
        
        T = max(dia_lens)
        if self.config.is_casual:
            attention_mask = []
            for length in dia_lens:
                mask = torch.tril(torch.ones(T, T, device=self.device))
                flat = torch.tensor(length * [1] + (T-length) * [0], device=self.device)
                mask *= flat.view(1, -1)
                # mask *= flat.view(-1, 1)
                attention_mask.append(mask)
            attention_mask = torch.stack(attention_mask, dim=0)
        else:
            attention_mask = torch.tensor([length * [1] + (T-length) * [0] for length in dia_lens], device=self.device)
        
        padded_inputs = torch.stack([torch.stack(inp + (T-len(inp)) * (inp[0].new_zeros(self.hidden_size),)) for inp in inputs])

        # (B, T, H)
        hidden_states = padded_inputs
        for layer in self.transformer:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states

    def get_hidden_size(self):
        return self.hidden_size


class SparseTransformerDM(nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()

        self.hf_model_name = hf_model_name

        self.model = MPNetModel.from_pretrained(hf_model_name)
        self.tokenizer = MPNetTokenizer.from_pretrained(hf_model_name)

    def get_hparams(self):
        return {"hf_model_name": self.hf_model_name}

    @property
    def device(self):
        return self.model.device
    
    def _tokenize(self, batch, device, padding_idx=1):  # padding_idx that is used in MPNet
        # group utterances by turns in order to tokenize and pad them jointly
        uts_grouped_by_turn = defaultdict(list)
        max_n_utterances = max(len(dia) for dia in batch)

        for dia in batch:
            for i, item in enumerate(dia):
                uts_grouped_by_turn[i].append(f"[{item['speaker']}] {item['utterance']}")
            for i in range(len(dia), max_n_utterances):
                uts_grouped_by_turn[i].append('')

        uts_grouped_by_turn_tokenized = [None for _ in range(max_n_utterances)]
        uts_lens = [None for _ in range(max_n_utterances)]
        for i, uts in uts_grouped_by_turn.items():
            tokens = self.tokenizer(uts, padding='longest', return_tensors='pt')
            uts_grouped_by_turn_tokenized[i] = tokens
            uts_lens[i] = tokens['input_ids'].shape[1]
        
        input_ids = torch.cat([group['input_ids'] for group in uts_grouped_by_turn_tokenized], dim=1)

        # make extended attention mask of size (B, T, T)
        attention_mask = []
        for i in range(len(batch)):
            masks_per_utterance = []
            for j, group in enumerate(uts_grouped_by_turn_tokenized):
                mask = group['attention_mask'][i]
                T = uts_lens[j]
                masks_per_utterance.append(mask[None, :].expand(T, T))
            attention_mask.append(torch.block_diag(*masks_per_utterance))
        attention_mask = torch.stack(attention_mask, dim=0)
        
        # allow CLS tokens attend to each other
        extended_attention_mask = attention_mask
        for i in range(len(uts_lens)):
            cls_idx = sum(uts_lens[:i])
            extended_attention_mask[:, :, cls_idx] = 1
        
        # assign positions within each utterance
        position_ids = []
        for group in uts_grouped_by_turn_tokenized:
            ids = create_position_ids_from_input_ids(group['input_ids'], padding_idx=padding_idx)
            position_ids.append(ids)
        position_ids = torch.cat(position_ids, dim=1)
        
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": extended_attention_mask.to(device),
            "position_ids": position_ids.to(device)
        }
        
        return inputs, uts_lens
    
    def forward(self, batch):
        device = self.device

        inputs, uts_lens = self._tokenize(batch, device)
        outputs = self.model(**inputs)

        hidden_states = []
        for i in range(len(uts_lens)):
            # take final representation of <s> token
            j = sum(uts_lens[:i])
            hidden_states.append(outputs.last_hidden_state[:, j, :])
        
        # (B, T, H)
        hidden_states = torch.stack(hidden_states, dim=1)

        return hidden_states

    def get_hidden_size(self):
        return self.model.config.hidden_size


class HSSADM(nn.Module):
    def __init__(self, hf_model_name, config: HSSAConfig):
        super().__init__()

        self.hf_model_name = hf_model_name
        self.config = config

        config.pool_utterances = True
        self.model = HSSAModel.from_pretrained(hf_model_name, config=config)
        self.tokenizer = HSSATokenizer.from_pretrained(hf_model_name)
    
    @property
    def device(self):
        return self.model.device

    def forward(self, batch):
        tokenized = self.tokenizer(batch).to(self.device)
        hidden_states = self.model(**tokenized)
        return hidden_states
    
    def get_hidden_size(self):
        return self.config.hidden_size