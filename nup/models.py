from torch.utils.data import Dataset
import math
import json
from typing import List, Literal, Union
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from bisect import bisect_left
import os
from collections import defaultdict

# os.chdir('nup')

class NUPDataset(Dataset):
    chunk_size = 2048
    def __init__(self, path, split: Literal['train', 'test', 'val'], fraction=1.):
        self.split = split
        self.path = path

        if split == 'train':
            max_n_chunks = 2556
        elif split == 'test' or split == 'val':
            max_n_chunks = 142

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
        item = json.load(open(f'{self.path}/dataset/{self.split}/{i_chunk}.json', 'r'))[idx_within_chunk]
        return item


class DialogueDataset(NUPDataset):
    def __getitem__(self, i):
        item = super().__getitem__(i)
        return item['context'] + [item['target']]


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

    def forward(self, sentences: List[str]) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
        input = self.tokenizer(sentences, padding='longest', return_tensors='pt')
        output = self.model(input_ids=input['input_ids'].to(self.model.device))
        
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


LR = 5e-4
WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.999)
BATCH_SIZE = 32
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


class ChainCosine(nn.Module):
    def __init__(self, encoder_name, projection_size, context_size, tau, finetune_encoder_layers: int = 0):
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

    def get_logits(self, batch):
        # collate utterances to list and get sentence encodings
        utterances = []
        rle = []
        context_speaker = []
        target_speaker = []
        for item in batch:
            cur_utterances = [ut['utterance'] for ut in item['context']]+[item['target']['utterance']]
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
            end = start + length - 1

            context_encoding = encodings[end-2]
            sec = start-1 if self.context_size is None else max(start, end-2-self.context_size)
            for enc in encodings[end-3:sec:-1]:
                context_encoding = (1 - self.tau) * context_encoding + enc * self.tau
            context_encoding = torch.concat([encodings[end-1], context_encoding, context_encoding.new_tensor([context_speaker[i]])])

            target_encoding = torch.concat([encodings[end], context_encoding.new_tensor([target_speaker[i]])])
            
            context_batch.append(context_encoding)
            target_batch.append(target_encoding)
        
        # append speaker embeddings
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
        self.ranker.bias.data.zero_()
    
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.ranker(x)
        return x.squeeze(-1)


def normalized_inversions_number(arr):
    """Function to count number of inversions in a permutation of 0, 1, ..., n-1."""
    n = len(arr)
    v = list(range(n))
    ans = 0
    for i in range(n):
        itr = bisect_left(v, arr[i])-1
        ans += itr
        del v[itr]
    max_inversions = n * (n - 1) // 2
    return ans / max_inversions


class UtteranceSorter(nn.Module):
    def __init__(self, config: TransformerConfig, encoder_name, dropout_prob, finetune_encoder_layers: int = 0):
        super().__init__()

        self.encoder = mySentenceTransformer(encoder_name)
        self.encoder.requires_grad_(False)
        freeze_hf_model(self.encoder.model, finetune_encoder_layers)

        sentence_embedding_dimension = self.encoder.model.config.hidden_size
        self.hidden_size = sentence_embedding_dimension // 2
        
        self.projector = Projector(sentence_embedding_dimension, self.hidden_size)
        config.hidden_size = self.hidden_size
        config.intermediate_size = 4 * self.hidden_size
        self.transformer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ranker_head = RankerHead(self.hidden_size, dropout_prob)

    def get_logits(self, batch):
        device = self.encoder.model.device

        inputs = []
        for dia in batch:
            utterances = self.projector(self.encoder([f"[{item['speaker']}] {item['utterance']}" for item in dia]))
            encoded_dialogue = torch.unbind(utterances)
            inputs.append(encoded_dialogue)
        
        T = max(len(inp) for inp in inputs)
        attention_mask = torch.BoolTensor([len(inp) * [False] + (T-len(inp)) * [True] for inp in inputs]).to(device)
        padded_inputs = torch.stack([torch.stack(inp + (T-len(inp)) * (inp[0].new_zeros(self.hidden_size),)) for inp in inputs])

        # (B, T, H)
        hidden_states = padded_inputs
        for layer in self.transformer:
            hidden_states = layer(hidden_states, attention_mask)

        # (B, T)
        return self.ranker_head(hidden_states)

    def get_permutaions(self, ranks_logits, dia_lens):
        permutations = torch.argsort(ranks_logits, dim=1).detach().cpu().tolist()
        return [perm[:length] for perm, length in zip(permutations, dia_lens)]
    
    def forward(self, batch):
        device = self.encoder.model.device

        ranks_logits = self.get_logits(batch)
        ranks_probs = F.softmax(ranks_logits, dim=1)
        
        B, T = ranks_logits.shape
        ranks_probs_true = torch.linspace(0, 1, T, device=device).unsqueeze(0).expand(B, T)

        loss = F.cross_entropy(ranks_probs, ranks_probs_true, reduction='mean')
        permutations = self.get_permutaions(ranks_logits, [len(dia) for dia in batch])
        sorting_index = 1-np.mean([normalized_inversions_number(perm) for perm in permutations])

        return loss, sorting_index

    def augment(self, batch):
        ranks_logits = self.get_logits(batch)
        permutations = self.get_permutaions(ranks_logits, [len(dia) for dia in batch])

        return [[dia[i] for i in perm] for dia, perm in zip(batch, permutations)]

    @staticmethod
    def from_checkpoint(path_to_ckpt, map_location=None, **kwargs):
        return Learner.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=UtteranceSorter(**kwargs)
        ).model


class UtteranceSorter2(nn.Module):
    def __init__(self, hf_model_name, dropout_prob, finetune_encoder_layers: int = 1):
        super().__init__()

        self.model = AutoModel.from_pretrained(hf_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)    
        freeze_hf_model(self.model, finetune_encoder_layers)

        self.ranker_head = RankerHead(self.model.config.hidden_size, dropout_prob)
    
    def _tokenize(self, batch, device):
        # group utterances by turns in order to tokenize and pad them jointly
        uts_grouped_by_turn = defaultdict(list)
        dia_lens = [len(dia) for dia in batch]
        max_n_utterances = max(dia_lens)

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
        
        return {"input_ids": input_ids.to(device), "attention_mask": extended_attention_mask.to(device)}, uts_lens, dia_lens
    
    def forward(self, batch):
        device = self.model.device

        inputs, uts_lens, dia_lens = self._tokenize(batch, device)
        outputs = self.model(**inputs)

        hidden_states = []
        for i in range(len(uts_lens)):
            j = sum(uts_lens[:i])
            hidden_states.append(outputs.last_hidden_state[:, j, :])
        
        # (B, T, H)
        hidden_states = torch.stack(hidden_states, dim=1)
        B, T, _ = hidden_states.shape

        # (B, T)
        ranks_logits = self.ranker_head(hidden_states)
        dia_lens_tensor = torch.tensor(dia_lens, device=device)[:, None]
        is_padded_utterance = torch.arange(T, device=device)[None, :].expand(B, T) > dia_lens_tensor
        ranks_logits = ranks_logits.masked_fill(is_padded_utterance, -torch.inf)

        ranks_probs = F.softmax(ranks_logits, dim=1)

        ranks_probs_true = torch.linspace(0, 1, T, device=device).unsqueeze(0).expand(B, T)

        loss = F.cross_entropy(ranks_probs, ranks_probs_true, reduction='mean')
        predicted_permutations = torch.argsort(ranks_probs, dim=1).detach().cpu().tolist()
        predicted_permutations = [perm[:length] for perm, length in zip(predicted_permutations, dia_lens)]
        sorting_index = 1-np.mean([normalized_inversions_number(perm) for perm in predicted_permutations])

        return loss, sorting_index


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
        self.logger.log_hyperparams(self.optimizers().defaults)

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
        return optimizer


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from lightning.pytorch.callbacks import ModelCheckpoint
    from datetime import datetime
    torch.set_float32_matmul_precision('medium')
    import os
    from functools import partial

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', dest='model', required=True, choices=['chainer', 'sorter', 'sorter-2'])
    ap.add_argument('--name', dest='name', required=True)
    args = ap.parse_args()


    dataset = NUPDataset if args.model == 'chainer' else DialogueDataset

    train_loader = DataLoader(
        dataset=dataset('.', 'train', fraction=1),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=dataset('.', 'val', fraction=0.1),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=10,
        collate_fn=collate_fn
    )

    if args.model == 'chainer':
        model = ChainCosine(
            encoder_name='sentence-transformers/all-mpnet-base-v2',
            projection_size=PROJECTION_SIZE,
            context_size=6,
            tau=0.5,
            finetune_encoder_layers=1
        )
    elif args.model == 'sorter':
        model = UtteranceSorter(
            config=TransformerConfig(
                hidden_size=None,
                num_attention_heads=4,
                attention_probs_dropout_prob=0.05,
                intermediate_size=None,
                n_layers=4
            ),
            encoder_name='sentence-transformers/all-mpnet-base-v2',
            dropout_prob=0.05,
            finetune_encoder_layers=1
        )
    else:
        model = UtteranceSorter2(
            hf_model_name='sentence-transformers/all-mpnet-base-v2',
            dropout_prob=0.05,
            finetune_encoder_layers=1
        )

    learner = Learner(model)
    learner.configure_optimizers = partial(learner.configure_optimizers, config={'lr': LR, 'weight_decay': WEIGHT_DECAY, 'betas': BETAS})

    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric',
        save_last=True,
        save_top_k=3,
        mode='max',
    )

    # logger = pl.loggers.TensorBoardLogger(
    #     save_dir='.',
    #     version=args.name,
    #     name='lightning_logs'
    # )

    trainer = pl.Trainer(
        # max_epochs=1,
        max_time={'minutes': 1},

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        precision="16-mixed",

        # logging and checkpointing
        # val_check_interval=500,
        # check_val_every_n_epoch=
        logger=True,
        enable_progress_bar=False,
        profiler=None,
        callbacks=[checkpoint_callback],

        # check if model is implemented correctly
        overfit_batches=1,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # do magic!
    trainer.fit(learner, train_loader, val_loader)

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))