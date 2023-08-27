from torch.utils.data import Dataset
from math import ceil
import json
from typing import Literal


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
        i_chunk = ceil(i / self.n_chunks)
        idx_within_chunk = i % self.chunk_size
        item = json.load(open(f'nup-dataset/{self.split}/{i_chunk}.json', 'r'))[idx_within_chunk]
        return item

def collate_fn(batch):
    return batch


import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np

LR = 1e-3
WEIGHT_DECAY = 1e-2
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
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        else:
            x = x.to(torch.float32)
        x = x.cuda()
        x = x + F.gelu(self.linear_1(self.norm1(x)))
        x = x + F.gelu(self.linear_2(self.norm2(x)))

        return F.normalize(self.final(x), dim=1)


class ChainCosine(pl.LightningModule):
    def __init__(self, n_speakers, projection_size, tau):
        super().__init__()

        self.n_speakers = n_speakers
        self.projection_size = projection_size
        self.tau = tau
        
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.encoder.requires_grad_(False)
        
        sentence_embedding_dimension = self.encoder.get_sentence_embedding_dimension()
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
    
    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
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
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from lightning.pytorch.callbacks import ModelCheckpoint
    from datetime import datetime
    torch.set_float32_matmul_precision('medium')
    import os

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


    train_loader = DataLoader(
        dataset=NUPDataset('train', fraction=1),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=NUPDataset('val', fraction=1),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=10,
        collate_fn=collate_fn
    )

    model = ChainCosine(n_speakers=2, projection_size=PROJECTION_SIZE, tau=0.3)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_last=True,
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        # max_epochs=100,
        max_time={'minutes': 120},

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
        overfit_batches=None,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # do magic!
    trainer.fit(model, train_loader, val_loader)

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))