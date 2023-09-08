from torch.utils.data import Dataset
import math
import json
from typing import List, Literal
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, asdict
from bisect import bisect_left
import os
from collections import defaultdict
from transformers.models.mpnet.modeling_mpnet import create_position_ids_from_input_ids
from torch.optim.lr_scheduler import LambdaLR

# os.chdir('nup')

class NUPDataset(Dataset):
    chunk_size = 2048
    def __init__(self, path, split: Literal['train', 'test', 'val'], fraction=1.):
        self.split = split
        self.path = path

        if split == 'train':
            max_n_chunks = 2556
        elif split == 'test' or split == 'val':
            max_n_chunks = 141

        if isinstance(fraction, float):
            self.fraction = min(1., max(0., fraction))
            self.n_chunks = math.ceil(self.fraction * max_n_chunks)
        elif isinstance(fraction, int):
            self.fraction = min(max_n_chunks, max(1, fraction))
            self.n_chunks = fraction
        else:
            raise ValueError('fraction must be int or float')

        self.len = self.n_chunks * self.chunk_size
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one dialogue, represented with an object of the following schema:
        ```
        {
            "type": "object",
            "properties":
            {
                "context":
                {
                    "type": "array",
                    "items":
                    {
                        "type": "object",
                        "properties":
                        {
                            "utterance": {"type": "string"},
                            "speaker": {"type": "number"}
                        }
                    }
                },
                "target":
                {
                    "type": "object",
                    "properties":
                    {
                        "utterance": {"type": "string"},
                        "speaker": {"type": "number"}
                    }
                }
            }
        }
        ```"""
        i_chunk = math.floor(i / self.chunk_size)
        idx_within_chunk = i % self.chunk_size
        item = json.load(open(f'{self.path}/pairs/{self.split}/{i_chunk}.json', 'r'))[idx_within_chunk]
        return item


class DialogueDataset(Dataset):
    chunk_size = 512
    def __init__(self, path, split: Literal['train', 'test', 'val'], fraction=1.):
        self.split = split
        self.path = path

        if split == 'train':
            max_n_chunks = 880
        elif split == 'test' or split == 'val':
            max_n_chunks = 48

        if isinstance(fraction, float):
            self.fraction = min(1., max(0., fraction))
            self.n_chunks = math.ceil(self.fraction * max_n_chunks)
        elif isinstance(fraction, int):
            self.fraction = min(max_n_chunks, max(1, fraction))
            self.n_chunks = fraction
        else:
            raise ValueError('fraction must be int or float')

        self.len = self.n_chunks * self.chunk_size
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one dialogue, represented with an object of the following schema:
        ```
        {
            "type": "array",
            "items":
            {
                "type": "object",
                "properties":
                {
                    "utterance": {"type": "string"},
                    "speaker": {"type": "number"}
                }
            }
        }
        ```"""
        i_chunk = math.floor(i / self.chunk_size)
        idx_within_chunk = i % self.chunk_size
        item = json.load(open(f'{self.path}/dialogues/{self.split}/{i_chunk}.json', 'r'))[idx_within_chunk]
        return item


def collate_fn(batch):
    return batch


class mySentenceTransformer(nn.Module):
    """Imitation of SentenceTransformers (https://www.sbert.net/)"""

    def __init__(
            self,
            model_name='sentence-transformers/all-mpnet-base-v2',
            pooling=True
        ):
        """If `pooling=False`, then instead of sentence embeddings forward will return list of token embeddings."""
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling

        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, sentences: List[str]) -> List[torch.Tensor]:
        input = self.tokenizer(sentences, padding='longest', return_tensors='pt')
        output = self.model(
            input_ids=input['input_ids'].to(self.model.device),
            attention_mask=input['attention_mask'].to(self.model.device)
        )
        
        res = []
        for token_emb, attention in zip(output.last_hidden_state, input['attention_mask']):
            last_mask_id = len(attention)-1
            while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                last_mask_id -= 1
            embs = token_emb[:last_mask_id+1]
            if self.pooling:
                embs = torch.mean(embs, dim=0)
                embs = embs / torch.linalg.norm(embs)
            res.append(embs)

        return res


WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.999)
PROJECTION_SIZE = 512


class Projector(nn.Module):
    """Fully-Connected 2-layer Linear Model. Taken from linking prediction paper code."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, input_size)
        self.linear_2 = nn.Linear(input_size, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.final = nn.Linear(input_size, output_size)
        # self.orthogonal_initialization()

    def orthogonal_initialization(self):
        for l in [self.linear_1, self.linear_2]:
            torch.nn.init.orthogonal_(l.weight)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.stack(x)
        else:
            x = x.to(torch.float32)
        x = x.cuda()
        x = x + F.gelu(self.linear_1(self.norm1(x)))
        x = x + F.gelu(self.linear_2(self.norm2(x)))

        return F.normalize(self.final(x), dim=1)


class LightningCkptLoadable:
    @staticmethod
    def from_checkpoint(path_to_ckpt, model, map_location=None):
        return Learner.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=model
        ).model


class ChainCosine(nn.Module, LightningCkptLoadable):
    def __init__(self, encoder_name, projection_size, context_size, tau, finetune_encoder_layers: int = 0):
        """
        Params
        ------
        - `encoder_name`: name of underlying hf model for sentence embedding
        - `projection_size`: final representation size
        - `context_size`: number of recent utterances that are encoded into contextual representation, if None then use full context
        - `tau`: EMA coefficient for context encoding
        - `finetune_encoder_layers`: number of last layers of hf model to finetune
        """
        super().__init__()

        self.projection_size = projection_size
        self.tau = tau
        
        self.encoder = mySentenceTransformer(encoder_name, pooling=True)
        freeze_hf_model(self.encoder.model, finetune_encoder_layers)

        self.sentence_embedding_dimension = self.encoder.model.config.hidden_size
        self.context_size = context_size
        self.context_projector = Projector(
            input_size=self.sentence_embedding_dimension*2+1,
            output_size=self.projection_size
        )
        self.target_projector = Projector(
            input_size=self.sentence_embedding_dimension+1,
            output_size=self.projection_size
        )
    
    def get_hparams(self):
        return {
            "projection_size": self.projection_size,
            "tau": self.tau,
            "sentence_embedding_dimension": self.sentence_embedding_dimension,
            "context_size": self.context_size
        }

    def get_logits(self, batch):
        # collate utterances to list and get sentence encodings
        utterances = []
        rle = []
        context_speaker = []
        target_speaker = []
        if self.context_size is None:
            context_slice = slice(None, None, -1)
        else:
            context_slice = slice(-1, -self.context_size-1, -1)
        for item in batch:
            cur_utterances = [item['target']['utterance']] + [ut['utterance'] for ut in item['context'][context_slice]]
            utterances.extend(cur_utterances)
            rle.append(len(cur_utterances))
            context_speaker.append(item['context'][-1]['speaker'])
            target_speaker.append(item['target']['speaker'])
        
        encodings = self.encoder(utterances)

        # collate context and target encodings
        context_batch = []
        target_batch = []
        for i, length in enumerate(rle):
            start = sum(rle[:i])
            end = start + length

            if length == 2:
                context_encoding = torch.zeros_like(encodings[0])
            else:   # length > 2
                context_encoding = encodings[start+2]
            
            for enc in encodings[start+3:end]:
                context_encoding = (1 - self.tau) * context_encoding + enc * self.tau
            
            context_encoding = torch.concat([encodings[start+1], context_encoding, context_encoding.new_tensor([context_speaker[i]])])
            target_encoding = torch.concat([encodings[start], context_encoding.new_tensor([target_speaker[i]])])
            
            context_batch.append(context_encoding)
            target_batch.append(target_encoding)
        
        # project to joing embedding space
        context_batch = self.context_projector(
            torch.stack(context_batch) 
        )
        target_batch = self.target_projector(
            torch.stack(target_batch)
        )

        return context_batch @ target_batch.T

    def forward(self, batch):
        logits = self.get_logits(batch)
        labels = torch.arange(len(batch), device='cuda')
        
        loss = F.cross_entropy(logits, labels, reduction='mean')
        accuracy = (torch.argmax(logits, dim=1) == labels).float().mean().item()

        return loss, accuracy
    
    @torch.no_grad()
    def score(self, dialogue):
        batch = []
        for i in range(1, len(dialogue)):
            batch.append({
                'context': dialogue[:i],
                'target': dialogue[i]
            })
        B = len(batch)
        logits = self.get_logits(batch)
        return F.softmax(logits, dim=1).diag().log10().mean().cpu().item()

    @staticmethod
    def from_checkpoint(path_to_ckpt, map_location=None, **kwargs):
        return Learner.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=ChainCosine(**kwargs)
        ).model


def freeze_hf_model(hf_model, finetune_encoder_layers: int):
        """Freeze all encoder layers except last `finetune_encoder_layers`"""
        hf_model.requires_grad_(False)
        n_layers = hf_model.config.num_hidden_layers
        for i in range(n_layers):
            hf_model.encoder.layer[i].requires_grad_(i>=n_layers-finetune_encoder_layers)


@dataclass
class TransformerConfig:
    hidden_size: int
    num_attention_heads: int
    attention_probs_dropout_prob: float
    intermediate_size: int
    n_layers: int


class SelfAttention(nn.Module):
    def __init__(
            self,
            config: TransformerConfig
        ):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.config = config

        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.norm = nn.LayerNorm(config.hidden_size)
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        change view from (B, T, H) to (B, n, T, h)
        - B batch size
        - T longest sequence size
        - H hidden size
        - n number of att heads
        - h single att head size
        """
        new_x_shape = x.size()[:-1] + (self.config.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask):
        """
        x: (B, T, H)
        attention_mask: (B, T), BoolTensor, if True, then ignore corresponding token
        """
        # (B, T, H)
        hidden_states = self.norm(x)

        # (B, T, H)
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        # (B, n, T, h)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # (B, n, T, T)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_scores = attention_scores.masked_fill(attention_mask[:, None, None, :], -torch.inf)

        # (B, n, T, T)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # (B, n, T, h)
        c = torch.matmul(attention_probs, v)

        # (B, T, H)
        c = c.permute(0, 2, 1, 3).contiguous()
        new_c_shape = c.size()[:-2] + (self.config.hidden_size,)
        c = c.view(*new_c_shape)

        # (B, T, H)
        return x + self.o(c)


class FFBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.nonlinear = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, x):
        return x + self.linear2(self.nonlinear(self.linear1(self.norm(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.att = SelfAttention(config)
        self.ff = FFBlock(config)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attention_mask):
        x = self.att(x, attention_mask)
        x = self.ff(x)
        return self.norm(x)


class RankerHead(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.ranker = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.ranker(x)
        return x.squeeze(-1)


class BaseUtteranceSorter(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        # self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def augment(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        ranks_logits = self.get_logits(batch)
        mask = self._make_attention_mask(dia_lens, device)
        unbinded_ranks_logits = self._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self._to_permutations(unbinded_ranks_logits)

        return [[dia[i] for i in perm] for dia, perm in zip(batch, permutations)]

    def forward(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        ranks_logits = self.get_logits(batch)

        # zero attention to padding token-utterances
        mask = self._make_attention_mask(dia_lens, device)
        ranks_logits.masked_fill_(mask, -torch.inf)

        # calculate loss
        _, T = ranks_logits.shape
        ranks_true = self._make_true_ranks(T, dia_lens, device)
        loss = self.loss_fn(ranks_logits, ranks_true)

        # calculate metric
        unbinded_ranks_logits = self._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self._to_permutations(unbinded_ranks_logits)
        sorting_index = 1-np.mean([self._normalized_inversions_count(perm) for perm in permutations])

        return loss, sorting_index

    @staticmethod
    def _make_true_ranks(T, dia_lens, device):
        res = []

        def sigmoid(x):
            """sigmoid for x in range [0, 1]"""
            return 1 / (1 + (2 * x) ** 5)
        
        for length in dia_lens:
            ranks = torch.linspace(0, 1, length, device=device)
            padded_ranks = F.pad(ranks, pad=(0, T-length), value=1)
            shaped_ranks = sigmoid(padded_ranks)
            res.append(shaped_ranks)
        
        return torch.stack(res)

    @staticmethod
    def _to_permutations(unbinded_ranks_logits):
        """permutations with respect to descending order"""
        return [logits.argsort(descending=True) for logits in unbinded_ranks_logits]

    @staticmethod
    def _make_attention_mask(dia_lens, device):
        """this mask indicates padding tokens(utterances)"""
        T = max(dia_lens)
        dia_lens_expanded = torch.tensor(dia_lens, device=device)[:, None]
        max_dia_len_expanded = torch.arange(T, device=device)[None, :]
        return dia_lens_expanded <= max_dia_len_expanded
    
    @staticmethod
    def _unbind_logits(logits, mask, dia_lens):
        """get list of tensors with logits corresponding to tokens(utterances) that are not padding ones only"""
        return logits[~mask].detach().cpu().split(dia_lens)

    @staticmethod
    def _normalized_inversions_count(arr):
        """Function to count number of inversions in a permutation of 0, 1, ..., n-1."""
        n = len(arr)
        v = list(range(n))
        ans = 0
        for i in range(n):
            itr = bisect_left(v, arr[i])
            ans += itr
            del v[itr]
        max_inversions = n * (n - 1) / 2
        return ans / max_inversions


class UtteranceSorter(BaseUtteranceSorter, nn.Module, LightningCkptLoadable):
    def __init__(self, config: TransformerConfig, encoder_name, dropout_prob, finetune_encoder_layers: int = 0):
        super().__init__()

        self.encoder_name = encoder_name
        self.dropout_prob = dropout_prob
        self.finetune_encoder_layers = finetune_encoder_layers

        self.encoder = mySentenceTransformer(encoder_name)
        self.encoder.requires_grad_(False)
        freeze_hf_model(self.encoder.model, finetune_encoder_layers)

        sentence_embedding_dimension = self.encoder.model.config.hidden_size
        self.hidden_size = sentence_embedding_dimension // 2

        self.speaker_embeddings = nn.Embedding(2, 8)
        
        self.projector = Projector(sentence_embedding_dimension, self.hidden_size - 8)
        config.hidden_size = self.hidden_size
        config.intermediate_size = 4 * self.hidden_size
        self.transformer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ranker_head = RankerHead(self.hidden_size, dropout_prob)

        self.config = config

    def get_hparams(self):
        res = {
            "encoder_name": self.encoder_name,
            "head_dropout_prob": self.dropout_prob,
            "finetune_encoder_layers": self.finetune_encoder_layers,
            "hidden_size": self.hidden_size
        }
        res.update(asdict(self.config))
        return res

    @property
    def device(self):
        return self.encoder.model.device

    def get_logits(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        inputs = []
        for dia in batch:
            speaker_ids = torch.tensor([item['speaker'] for item in dia], device=device, dtype=torch.long)
            speaker_embeddings = self.speaker_embeddings(speaker_ids)

            sentence_embeddings = self.encoder([item['utterance'] for item in dia])
            sentence_embeddings = self.projector(sentence_embeddings)
            
            utterance_embeddings = torch.cat([sentence_embeddings, speaker_embeddings], dim=1)
            utterance_embeddings = torch.unbind(utterance_embeddings)
            inputs.append(utterance_embeddings)
        
        T = max(dia_lens)
        attention_mask = torch.BoolTensor([length * [False] + (T-length) * [True] for length in dia_lens]).to(device)
        padded_inputs = torch.stack([torch.stack(inp + (T-len(inp)) * (inp[0].new_zeros(self.hidden_size),)) for inp in inputs])

        # (B, T, H)
        hidden_states = padded_inputs
        for layer in self.transformer:
            hidden_states = layer(hidden_states, attention_mask)

        # (B, T)
        return self.ranker_head(hidden_states)


class UtteranceSorter2(BaseUtteranceSorter, nn.Module, LightningCkptLoadable):
    def __init__(self, hf_model_name, dropout_prob, finetune_encoder_layers: int = 1):
        super().__init__()

        self.hf_model_name = hf_model_name
        self.dropout_prob = dropout_prob
        self.finetune_encoder_layers = finetune_encoder_layers

        self.model = AutoModel.from_pretrained(hf_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)    
        freeze_hf_model(self.model, finetune_encoder_layers)

        self.ranker_head = RankerHead(self.model.config.hidden_size, dropout_prob)
    
    def get_hparams(self):
        return {
            "hf_model_name": self.hf_model_name,
            "head_dropout_prob": self.dropout_prob,
            "finetune_encoder_layers": self.finetune_encoder_layers,
        }

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
    
    def get_logits(self, batch):
        device = self.device

        inputs, uts_lens = self._tokenize(batch, device)
        outputs = self.model(**inputs)

        hidden_states = []
        for i in range(len(uts_lens)):
            j = sum(uts_lens[:i])
            hidden_states.append(outputs.last_hidden_state[:, j, :])
        
        # (B, T, H)
        hidden_states = torch.stack(hidden_states, dim=1)

        # (B, T)
        return self.ranker_head(hidden_states)


class Learner(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        loss, metric = self.forward(batch)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=BATCH_SIZE
        )
        self.log(
            name='train_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=BATCH_SIZE
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, metric = self.forward(batch)
        self.log(
            name='val_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=BATCH_SIZE
        )
        self.log(
            name='val_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=BATCH_SIZE
        )
    
    def on_train_start(self):
        optim_hparams = self.optimizers().defaults
        model_hparams = self.model.get_hparams()
        model_hparams.update(optim_hparams)
        self.logger.log_hyperparams(model_hparams)

    def configure_optimizers(self, config):
        """Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=config['lr'], betas=config['betas'])
        def lr_foo(step, warmup_steps=200):
            step = step % warmup_steps
            lr_scale = (step + 1) / warmup_steps
            return lr_scale

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from datetime import datetime
    torch.set_float32_matmul_precision('medium')
    import os
    from functools import partial

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', dest='model', required=True, choices=['pairwise', 'listwise', 'listwise2'])
    ap.add_argument('--name', dest='name', required=True)
    args = ap.parse_args()

    if args.model == 'pairwise':
        model = ChainCosine(
            encoder_name='sentence-transformers/all-mpnet-base-v2',
            projection_size=PROJECTION_SIZE,
            context_size=6,
            tau=0.5,
            finetune_encoder_layers=1
        )
        LR = 5e-4
        BATCH_SIZE = 64
    elif args.model == 'listwise':
        model = UtteranceSorter(
            config=TransformerConfig(
                hidden_size=None,
                num_attention_heads=4,
                attention_probs_dropout_prob=0.02,
                intermediate_size=None,
                n_layers=4
            ),
            encoder_name='sentence-transformers/all-mpnet-base-v2',
            dropout_prob=0.02,
            finetune_encoder_layers=1
        )
        LR = 3e-6
        BATCH_SIZE = 512
    elif args.model == 'listwise2':
        # exit()
        model = UtteranceSorter2(
            hf_model_name='sentence-transformers/all-mpnet-base-v2',
            dropout_prob=0.,
            finetune_encoder_layers=1
        )
        LR = 1e-5
        BATCH_SIZE = 32

    learner = Learner(model)
    learner.configure_optimizers = partial(learner.configure_optimizers, config={'lr': LR, 'weight_decay': WEIGHT_DECAY, 'betas': BETAS})

    dataset = NUPDataset if args.model == 'pairwise' else DialogueDataset

    train_loader = DataLoader(
        dataset=dataset('.', 'train', fraction=1.),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=dataset('.', 'val', fraction=.5),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric',
        save_last=True,
        save_top_k=3,
        mode='max',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = pl.loggers.TensorBoardLogger(
        save_dir='.',
        version=args.name,
        name='logs/training'
    )

    trainer = pl.Trainer(
        # max_epochs=1,
        max_time={'hours': 24},
        
        # max_time={'minutes': 2},
        # max_steps=30,

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        precision="16-mixed",

        # logging and checkpointing
        val_check_interval=400,
        # check_val_every_n_epoch=1,
        logger=logger,
        enable_progress_bar=False,
        profiler=None,
        callbacks=[checkpoint_callback, lr_monitor],
        # log_every_n_steps=1,

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # do magic!
    trainer.fit(
        learner, train_loader, val_loader,
        # ckpt_path='logs/training/listwise-sigmoid/checkpoints/last.ckpt'
    )

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))