from torch.utils.data import Dataset
from math import ceil
import json
from typing import List, Literal, Union
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


class NUPDataset(Dataset):
    chunk_size = 2048
    def __init__(self, split: Literal['train', 'test', 'val'], fraction=1):
        self.fraction = min(1, max(0, fraction))
        self.split = split

        if split == 'train':
            max_n_chunks = 2801
        elif split == 'test' or split == 'val':
            max_n_chunks = 155
            
        self.n_chunks = ceil(self.fraction * max_n_chunks)
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
        i_chunk = ceil(i / self.n_chunks)
        idx_within_chunk = i % self.chunk_size
        item = json.load(open(f'dataset/{self.split}/{i_chunk}.json', 'r'))[idx_within_chunk]
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

    def forward(self, sentences: List[str]) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
        input = self.tokenizer(sentences, padding='longest', return_tensors='pt')
        output = self.model(**input)
        
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


LR = 1e-3
WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.999)
BATCH_SIZE = 64
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
        self.orthogonal_initialization()

    def orthogonal_initialization(self):
        for l in [self.linear_1, self.linear_2]:
            torch.nn.init.orthogonal_(l.weight)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        else:
            x = x.to(torch.float32)
        x = x.cuda()
        x = x + F.gelu(self.linear_1(self.norm1(x)))
        x = x + F.gelu(self.linear_2(self.norm2(x)))

        return F.normalize(self.final(x), dim=1)


class ChainCosine(nn.Module):
    def __init__(self, encoder_name, n_speakers, projection_size, tau, finetune_encoder=False):
        super().__init__()

        self.n_speakers = n_speakers
        self.projection_size = projection_size
        self.tau = tau
        
        self.encoder = mySentenceTransformer(encoder_name, pooling=True)
        self.encoder.requires_grad_(finetune_encoder)

        sentence_embedding_dimension = self.encoder.model.config.hidden_size
        self.context_projector = Projector(
            input_size=sentence_embedding_dimension,
            output_size=self.projection_size
        )
        self.target_projector = Projector(
            input_size=sentence_embedding_dimension,
            output_size=self.projection_size
        )

        self.speaker_encoding = nn.Embedding(self.n_speakers, sentence_embedding_dimension)
    
    def forward(self, batch):
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
        
        encodings = self.encoder.encode(utterances, batch_size=16)

        # collate context and target encodings
        context_batch = []
        target_batch = []
        for i, length in enumerate(rle):
            start = sum(rle[:i])
            end = start + length
            context_encoding = np.zeros_like(encodings[0])
            for i, enc in enumerate(encodings[end-2:start-1:-1]):
                context_encoding += enc * self.tau ** i 
            target_encoding = encodings[end-1]
            
            context_batch.append(context_encoding)
            target_batch.append(target_encoding)
        
        # append speaker embeddings
        context_batch = self.context_projector(
            torch.tensor(np.array(context_batch), device='cuda') + 
            self.speaker_encoding(torch.tensor(context_speaker, device='cuda'))
        )
        target_batch = self.target_projector(
            torch.tensor(np.array(target_batch), device='cuda') + 
            self.speaker_encoding(torch.tensor(target_speaker, device='cuda'))
        )

        # calculate loss
        logits = context_batch @ target_batch.T
        labels = torch.arange(len(batch), device='cuda')
        loss_c = F.cross_entropy(logits, labels, reduction='mean')
        loss_r = F.cross_entropy(logits.T, labels, reduction='mean')

        return (loss_c + loss_r) / 2


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

    def forward(self, x):
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
        
        self.att = SelfAttention(config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob)
        self.ff = FFBlock(config.hidden_size, config.intermediate_size)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = self.att(x)
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


class UtteranceRanker(nn.Module):
    def __init__(self, config: TransformerConfig, encoder_name, dropout_prob, finetune_encoder=False):
        super().__init__()

        self.encoder = mySentenceTransformer(encoder_name)
        self.encoder.requires_grad_(finetune_encoder)

        sentence_embedding_dimension = self.encoder.model.config.hidden_size
        self.hidden_size = sentence_embedding_dimension // 2
        
        self.projector = Projector(sentence_embedding_dimension, self.hidden_size)
        self.speaker_encoding = nn.Embedding(2, self.hidden_size)
        self.transformer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ranker_head = RankerHead(self.hidden_size, dropout_prob)
    
    def forward(self, batch):
        inputs = []
        for dia in batch:
            utterances = self.projector(self.encoder([item['utterance'] for item in dia]))
            speakers = self.speaker_encoding(torch.tensor([item['speaker'] for item in dia]))
            inputs.append(utterances + speakers)
        
        # padding and hence attention mask !!

        # (B, T, H)
        hidden_states = self.transformer(torch.tensor(inputs))

        # (B, T)
        ranks_logits = self.ranker_head(hidden_states)

        B, T = ranks_logits.shape

        ranks_probs = F.softmax(ranks_logits, dim=1)
        ranks_probs_true = torch.linspace(0, 1, T).unsqueeze(0).expand(B, T)  # may be instead of linspace make it exponential, quadratical etc?
        
        return F.cross_entropy(ranks_probs, ranks_probs_true, reduction='mean')


class Learner(pl.LightningModule):
    def __init__(self, model):
        self.model = model

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=BATCH_SIZE
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log(
            name='val_loss',
            value=loss,
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


    train_loader = DataLoader(
        dataset=NUPDataset('train', fraction=0.01),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=NUPDataset('val', fraction=0.01),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )

    model = ChainCosine(n_speakers=2, projection_size=PROJECTION_SIZE, tau=0.3)
    model.configure_optimizers = partial(model.configure_optimizers, config={'lr': LR, 'weight_decay': WEIGHT_DECAY, 'betas': BETAS})

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_last=True,
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=1,
        # max_time={'minutes': 120},

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        precision="16-mixed",

        # logging and checkpointing
        logger=True,
        enable_progress_bar=False,
        profiler=None,
        callbacks=[checkpoint_callback],

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # do magic!
    trainer.fit(model, train_loader, val_loader)

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))