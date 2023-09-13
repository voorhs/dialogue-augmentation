from torch.utils.data import Dataset
import math
import json
from typing import Literal, Tuple
import torch.nn as nn
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from bisect import bisect_left
import os
from torch.optim.lr_scheduler import LambdaLR
from aux import mySentenceTransformer, Projector
from dialoguemodels import UtteranceTransformerDMConfig, UtteranceTransformerDM, SparseTransformerDM, HSSADM, HSSAConfig


#### pairwise models ####


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


class LightningCkptLoadable:
    @staticmethod
    def from_checkpoint(path_to_ckpt, model, map_location=None):
        return Learner.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=model
        ).model


class ChainCosine(nn.Module, LightningCkptLoadable):
    def __init__(self, target_encoder, context_encoder, projection_size, context_size):
        super().__init__()

        self.projection_size = projection_size
        self.context_size = context_size

        self.target_encoder = target_encoder
        self.context_encoder = context_encoder
        
        self.context_projector = Projector(
            input_size=self.context_encoder.get_encoding_size(),
            output_size=self.projection_size
        )
        self.target_projector = Projector(
            input_size=self.target_encoder.get_encoding_size(),
            output_size=self.projection_size
        )

    def get_hparams(self):
        return {
            "context_size": self.context_size,
        }

    @property
    def device(self):
        return self.target_encoder.model.device
    
    def get_logits(self, batch):
        if self.context_size is None:
            context_slice = slice(None, None, None)
        else:
            context_slice = slice(-self.context_size, None, None)

        context_batch = []
        target_batch = []
        for pair in batch:
            context_batch.append(pair['context'][context_slice])
            target_batch.append(pair['target'])
        
        target_encodings = self.target_encoder(target_batch)
        context_encodings = self.context_encoder(context_batch)

        context_encodings = self.context_projector(context_encodings)
        target_encodings = self.target_projector(target_encodings)

        return context_encodings @ target_encodings.T

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
        logits = self.get_logits(batch)
        return F.softmax(logits, dim=1).diag().log10().mean().cpu().item()


class TargetEncoder(nn.Module):
    def __init__(self, sentence_encoder: mySentenceTransformer, n_speakers=2, speaker_embedding_dim=8):
        super().__init__()

        self.sentence_encoder = sentence_encoder
        self.n_speakers = n_speakers
        self.speaker_embedding_dim = speaker_embedding_dim

        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
    
    def forward(self, batch):
        uts = [item['utterance'] for item in batch]
        spe = [item['speaker'] for item in batch]
        
        sentence_embeddings = self.sentence_encoder(uts)
        speaker_ids = torch.tensor(spe, device=self.device)
        speaker_embeddings = self.speaker_embedding(speaker_ids)
        return torch.cat([torch.stack(sentence_embeddings), speaker_embeddings], dim=1)

    @property
    def device(self):
        return self.sentence_encoder.model.device
    
    def get_encoding_size(self):
        return self.speaker_embedding_dim + self.sentence_encoder.get_sentence_embedding_size()


class ContextEncoderConcat(nn.Module):
    def __init__(self, sentence_encoder: mySentenceTransformer, context_size):
        super().__init__()

        self.sentence_encoder = sentence_encoder
        self.context_size = context_size

    def forward(self, batch):
        uts = []
        lens = []
        for dia in batch:
            cur_uts = [item['utterance'] for item in dia]
            uts.extend(cur_uts)
            lens.append(len(cur_uts))
        
        sentence_embeddings = self.sentence_encoder(uts)
        d = self.sentence_encoder.get_sentence_embedding_size()
        res = []
        for i in range(len(batch)):
            start = sum(lens[:i])
            end = start + lens[i]
            n_zeros_to_pad = (self.context_size - lens[i]) * d
            enc = F.pad(torch.cat(sentence_embeddings[start:end]), pad=(n_zeros_to_pad, 0), value=0)
            res.append(enc)
        
        return res

    def get_encoding_size(self):
        return self.sentence_encoder.get_sentence_embedding_size() * self.context_size


class ContextEncoderEMA(nn.Module):
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


class ContextEncoderSparseTransformer(nn.Module):
    def __init__(self, hf_model_name, tau):
        super().__init__()
        self.tau = tau
        self.dialogue_model = SparseTransformerDM(hf_model_name)
    
    def forward(self, batch):
        hidden_states = self.dialogue_model(batch)
        
        lens = [len(context) for context in batch]
        d = self.dialogue_model.model.config.hidden_size
        
        sentence_embeddings = []
        for hs, length in zip(hidden_states, lens):
            sentence_embeddings.extend(torch.unbind(hs, dim=0)[:length])
        encodings = ContextEncoderEMA._ema(sentence_embeddings, lens, self.tau)
        
        return encodings

    def get_encoding_size(self):
        return 2 * self.dialogue_model.model.config.hidden_size


#### listwise models ####


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


class RankerHead(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.ranker = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.ranker(x)
        return x.squeeze(-1)


class UtteranceSorter(nn.Module):
    def __init__(self, dialogue_model, dropout_prob):
        super().__init__()

        self.dialogue_model = dialogue_model
        self.dropout_prob = dropout_prob
        
        self.ranker_head = RankerHead(dialogue_model.get_hidden_size(), dropout_prob)
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def get_hparams(self):
        res = {
            "head_dropout_prob": self.dropout_prob,
        }
        res.update(self.dialogue_model.get_hparams())
        return res

    @property
    def device(self):
        return self.dialogue_model.device

    def get_logits(self, batch):
        hidden_states = self.dialogue_model(batch)
        return self.ranker_head(hidden_states)

    def forward(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        ranks_logits = self.get_logits(batch)

        # zero attention to padding token-utterances
        mask = self._make_mask(dia_lens, device)
        ranks_logits.masked_fill_(mask, -1e4)

        # calculate loss
        ranks_logprobs = F.log_softmax(ranks_logits, dim=1)
        _, T = ranks_logits.shape
        ranks_true = self._make_true_ranks(T, dia_lens, device)
        loss = self.loss_fn(ranks_logprobs, ranks_true)

        # calculate metric
        unbinded_ranks_logits = self._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self._to_permutations(unbinded_ranks_logits)
        sorting_index = 1-np.mean([self._normalized_inversions_count(perm) for perm in permutations])

        return loss, sorting_index

    @torch.no_grad()
    def augment(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        ranks_logits = self.get_logits(batch)
        mask = self._make_mask(dia_lens, device)
        unbinded_ranks_logits = self._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self._to_permutations(unbinded_ranks_logits)

        return [[dia[i] for i in perm] for dia, perm in zip(batch, permutations)]

    @staticmethod
    def _make_true_ranks(T, dia_lens, device):
        res = []

        def sigmoid(x):
            """sigmoid for x in range [0, 1]"""
            return 1 / (1 + (2 * x) ** 5)
        
        for length in dia_lens:
            ranks = torch.linspace(1, 0, length, device=device)
            ranks = F.pad(ranks, pad=(0, T-length), value=0)
            # ranks = sigmoid(ranks)
            ranks = ranks / ranks.sum()
            res.append(ranks)
        
        return torch.stack(res)

    @staticmethod
    def _to_permutations(unbinded_ranks_logits):
        """permutations with respect to descending order"""
        return [logits.argsort(descending=True) for logits in unbinded_ranks_logits]

    @staticmethod
    def _make_mask(dia_lens, device):
        """this mask indicates padding tokens(utterances). used for ranking (not for transformer)"""
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


@dataclass
class LearnerConfig:
    kwargs: field(default_factory=dict)
    lr: float = None
    batch_size: int = None
    warmup_period: int = None
    do_periodic_warmup: bool = False
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)


class Learner(pl.LightningModule):
    def __init__(self, model, config: LearnerConfig):
        super().__init__()
        self.model = model
        self.config = config

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
            batch_size=self.config.batch_size
        )
        self.log(
            name='train_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size
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
            batch_size=self.config.batch_size
        )
        self.log(
            name='val_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
    
    def on_train_start(self):
        optim_hparams = self.optimizers().defaults
        model_hparams = self.model.get_hparams()
        model_hparams.update(optim_hparams)
        model_hparams['batch size'] = self.config.batch_size
        model_hparams['warmup period'] = self.config.warmup_period
        model_hparams['do periodic warmup'] = self.config.do_periodic_warmup
        model_hparams.update(self.config.kwargs)
        self.logger.log_hyperparams(model_hparams)

    def configure_optimizers(self):
        """Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        # blacklist_weight_modules = (NoneType,)   #(torch.nn.LayerNorm, torch.nn.Embedding)
        for pn, p in self.named_parameters():

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(pn)
            else:
                decay.add(pn)

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
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=self.config.betas)
        def lr_foo(step):
            warmup_steps = self.config.warmup_period
            periodic = self.config.do_periodic_warmup
            
            if warmup_steps is None:
                return 1
            if periodic:
                return (step % warmup_steps + 1) / warmup_steps
            else:
                return (step + 1) / warmup_steps if step < warmup_steps else 1

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

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from hssa.modeling_hssa import SegmentPooler
    def freeze_hssa(model: nn.Module, finetune_layers=0):
        model.embeddings.requires_grad_(False)
        model.embeddings.word_embeddings.weight[-2:].requires_grad_(True)

        model.encoder.requires_grad_(False)
        for i, layer in enumerate(model.encoder.layer):
            layer.requires_grad_(i>=model.config.num_hidden_layers-finetune_layers)

        for module in model.modules():
            if isinstance(module, SegmentPooler):
                module.requires_grad_(True)

    def freeze_hf_model(hf_model, finetune_layers):
        """Freeze all encoder layers except last `finetune_encoder_layers`"""
        hf_model.requires_grad_(False)
        n_layers = hf_model.config.num_hidden_layers
        for i in range(n_layers):
            hf_model.encoder.layer[i].requires_grad_(i>=n_layers-finetune_layers)


    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', dest='model', required=True, choices=[
        'pairwise-cat', 'pairwise-ema', 'pairwise-sparse-transformer',
        'listwise-utterance-transformer', 'listwise-sparse-transormer',
        'listwise-hssa'
    ])
    ap.add_argument('--name', dest='name', default=None)
    args = ap.parse_args()

    mpnet_name = 'sentence-transformers/all-mpnet-base-v2'

    if args.model == 'pairwise-cat':
        context_size = 6
        finetune_encoder_layers = 2

        encoder = mySentenceTransformer(mpnet_name)
        freeze_hf_model(encoder.model, finetune_encoder_layers)
        target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderConcat(encoder, context_size=context_size)
        model = ChainCosine(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=512,
            context_size=context_size
        )
        learner_config = LearnerConfig(
            batch_size=128,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=3e-6,
            kwargs={
                'context_size': context_size,
                'finetune_encoder_layers': finetune_encoder_layers
            }
        )
    elif args.model == 'pairwise-ema':
        context_size = 6
        finetune_encoder_layers = 2
        tau = 0.5

        encoder = mySentenceTransformer(mpnet_name)
        freeze_hf_model(encoder.model, finetune_encoder_layers)
        target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderEMA(encoder, context_size=context_size, tau=tau)
        model = ChainCosine(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=512,
            context_size=context_size
        )
        learner_config = LearnerConfig(
            batch_size=128,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=3e-6,
            kwargs={
                'context_size': context_size,
                'finetune_encoder_layers': finetune_encoder_layers,
                'tau': tau
            }
        )
    elif args.model == 'pairwise-sparse-transformer':
        context_size = 6
        finetune_encoder_layers = 1
        finetune_layers = 2
        tau = 0.5

        encoder = mySentenceTransformer(mpnet_name)
        freeze_hf_model(encoder.model, finetune_encoder_layers)
        target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderSparseTransformer(mpnet_name, tau=tau)
        freeze_hf_model(context_encoder.dialogue_model.model, finetune_layers)
        model = ChainCosine(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=512,
            context_size=context_size
        )
        learner_config = LearnerConfig(
            batch_size=128,
            warmup_period=200,
            do_periodic_warmup=True,
            lr=3e-6,
            kwargs={
                'context_size': context_size,
                'finetune_encoder_layers': finetune_encoder_layers,
                'finetune_layers': finetune_layers,
                'tau': tau
            }
        )
    elif args.model == 'listwise-utterance-transformer':
        ranker_head_dropout_prob = 0.02
        finetune_encoder_layers = 3
        config = UtteranceTransformerDMConfig(
            num_attention_heads=4,
            attention_probs_dropout_prob=0.02,
            n_layers=4,
            encoder_name='sentence-transformers/all-mpnet-base-v2',
            embed_turn_ids=False,
            is_casual=False
        )
        dialogue_model = UtteranceTransformerDM(config)
        freeze_hf_model(dialogue_model.encoder.model, finetune_encoder_layers)
        
        model = UtteranceSorter(
            dialogue_model=dialogue_model,
            dropout_prob=ranker_head_dropout_prob
        )
        learner_config = LearnerConfig(
            batch_size=192,
            warmup_period=200,
            do_periodic_warmup=False,
            lr=3e-6,
            kwargs={
                'dropout_prob': ranker_head_dropout_prob,
                'finetune_encoder_layers': finetune_encoder_layers
            }
        )
    elif args.model == 'listwise-sparse-transormer':
        ranker_head_dropout_prob = 0.02
        finetune_layers = 2
        dialogue_model = SparseTransformerDM(mpnet_name)
        freeze_hf_model(dialogue_model.model, finetune_layers)
        
        model = UtteranceSorter(
            dialogue_model=dialogue_model,
            dropout_prob=ranker_head_dropout_prob
        )
        learner_config = LearnerConfig(
            batch_size=32,
            warmup_period=200,
            do_periodic_warmup=False,
            lr=1e-5,
            kwargs={
                'dropout_prob': ranker_head_dropout_prob,
                'finetune_layers': finetune_layers
            }
        )
    elif args.model == 'listwise-hssa':
        ranker_head_dropout_prob = 0.02
        finetune_layers = 1
        config = HSSAConfig(
            max_turn_embeddings=20,
            casual_utterance_attention=False,
            pool_utterances=True
        )
        dialogue_model = HSSADM(mpnet_name, config)
        freeze_hssa(dialogue_model.model, 2)
        model = UtteranceSorter(
            dialogue_model=dialogue_model,
            dropout_prob=ranker_head_dropout_prob
        )
        learner_config = LearnerConfig(
            batch_size=32,
            warmup_period=200,
            do_periodic_warmup=True,
            lr=3e-6,
            kwargs={
                'dropout_prob': ranker_head_dropout_prob,
                'finetune_layers': finetune_layers
            }
        )

    # learner = Learner.load_from_checkpoint(
    #     # checkpoint_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/pairwise/checkpoints/last.ckpt',
    #     model=model
    # )
    learner = Learner(model, learner_config)

    dataset = NUPDataset if args.model.startswith('pairwise') else DialogueDataset
    def collate_fn(batch):
        return batch

    train_loader = DataLoader(
        dataset=dataset('.', 'train', fraction=1.),
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=dataset('.', 'val', fraction=.5),
        batch_size=learner_config.batch_size,
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
        # max_time={'hours': 24},
        
        max_time={'minutes': 5},
        # max_steps=30,

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        precision="16-mixed",

        # logging and checkpointing
        # val_check_interval=500,
        # check_val_every_n_epoch=1,
        logger=logger,
        enable_progress_bar=False,
        profiler=None,
        callbacks=[checkpoint_callback, lr_monitor],
        # log_every_n_steps=1,

        # check if model is implemented correctly
        overfit_batches=2,

        # check training_step and validation_step doesn't fail
        fast_dev_run=2,
        num_sanity_val_steps=2
    )

    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # do magic!
    trainer.fit(
        learner, train_loader, val_loader,
        # ckpt_path=
    )

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))