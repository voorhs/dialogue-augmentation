from .source_dataset import Dialogue
from pprint import pformat
from tqdm import tqdm
from random import shuffle
from torch.utils.data import Dataset
from typing import Literal, Union
import math
import json
import os


class ContextResponsePair:
    def __init__(
            self,
            context: Union[Dialogue, dict],
            response: Union[Dialogue, dict],
            idx: int = None,
            **fields
        ):
        """add any extra `fields` if extra info is needed to be saved"""

        if isinstance(context, Dialogue) and isinstance(response, Dialogue):
            self.content = {
                'context': context.content,
                'response': response.content
            }
            self.idx_within_source = context.idx_within_source
            self.idx = idx
        elif isinstance(context, dict) and isinstance(response, dict):
            self.content = {
                'context': context,
                'response': response
            }
        else:
            raise ValueError(f'context and response must be the same data type, got {type(context)} and {type(response)}')
        
        for key, val in fields.items():
            setattr(self, key, val)
    
    def asdict(self):
        return vars(self)
    
    def __repr__(self):
        return pformat(self.asdict(), indent=2)
    
    @staticmethod
    def from_dict(dct):
        return ContextResponsePair(
            context=dct['content']['context'],
            response=dct['content']['response'],
        )
    
    @staticmethod
    def load_train_sample(dct):
        return dct['content']



def make_pairs(dialogues):
    res = []
    for dia in tqdm(dialogues, desc='making pairs'):
        pairs = []
        for i in range(len(dia)-1):
            pairs.append((dia[:i+1], dia[i+1]))
        res.extend(pairs)
    shuffle(res)
    res = [ContextResponsePair(context=c, response=r, idx=i) for i, (c, r) in enumerate(res)]
    return res


class ContextResponseDataset(Dataset):
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
            raise ValueError('fraction must indicate number or fraction of chunks used (int or float)')

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
        
        chunk_name = f'{i_chunk}.json'
        chunk_path = os.path.join(self.path, self.split, chunk_name)
        dct = json.load(open(chunk_path, 'r'))[idx_within_chunk]
        
        return ContextResponsePair.load_train_sample(dct)
