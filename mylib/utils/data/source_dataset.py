import json
from random import shuffle, seed as set_seet
from math import ceil
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing import List, Tuple
from pprint import pformat
from copy import copy
from functools import partial
import os
from bisect import bisect_right
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple
import math
import json


class Dialogue:
    def __init__(
            self,
            utterances: List[str],
            speakers: List[int],
            source_dataset_name: str,
            idx_within_source: int,
            idx: int,
            **fields
        ):
        """add any extra `fields` if extra info is needed to be saved"""

        if len(utterances) != len(speakers):
            raise ValueError('`utterances` and `speakers` must be the same length')
        
        self.content = [
            {'utterance': ut, 'speaker': sp} for ut, sp in zip(utterances, speakers)
        ]
        
        self.source_dataset_name = source_dataset_name
        self.idx_within_source = idx_within_source
        self.idx = idx
        
        for key, val in fields.items():
            setattr(self, key, val)
    
    def asdict(self):
        return vars(self)
    
    def __repr__(self):
        return pformat(self.asdict(), indent=2)
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, idx_or_slice):
        res = copy(self)
        res.content = self.content[idx_or_slice]
        return res

    @staticmethod
    def get_train_sample(dct):
        return dct['content']


def parse_sample(
        raw_sample,
        tokenizer,
        bound=None,
        user_id=0,
        system_id=1,
    ) -> Tuple[List[str], List[str]]:
    
    if is_empty(raw_sample) or is_too_long(raw_sample) or has_only_single_utterance(raw_sample):
        return

    utterances = []
    speakers = []
    
    for turn in raw_sample:
        for sp, item in zip([user_id, system_id], ['user utterance', 'system response']):
            ut = turn[item]
            if ut == '':
                continue
            utterances.append(ut)
            speakers.append(sp)        
    
    if bound is not None and any(is_above_bound(ut, tokenizer, bound) for ut in utterances):
        # if there're any utterances with exceeding length
        return
    
    return utterances, speakers


def is_empty(dia):
    return len(dia) == 0


def is_too_long(dia):
    return len(dia) > 10


def has_only_single_utterance(dia):
    return len(dia) == 1 and (dia[0]['user utterance'] == '' or dia[0]['system response'] == '')


def is_above_bound(ut, tokenizer, bound):
    return len(tokenizer(ut)['input_ids']) > bound


def parse_dataset(dataset, name, tokenizer, bound):
    """iterates through `dataset` and parses dialogues that satisfy 2 conditions:
    - has from 2 to 20 utterances
    - has no utterances with more than `bound` tokens
    
    If dia satisfies conditions, it is converted to `Dialogue` data type."""
    res = []
    idx = 0

    fn = partial(parse_sample, tokenizer=tokenizer, bound=bound)
    parse_results = process_map(fn, dataset, max_workers=2, chunksize=300, desc=f'preprocessing {name}')
    for i, parsed_dia in enumerate(parse_results):
        if parsed_dia is None:
            continue
        utterances, speakers = parsed_dia
        dia = Dialogue(
            utterances=utterances,
            speakers=speakers,
            source_dataset_name=name,
            idx_within_source=i,
            idx=idx
        )
        idx += 1
        res.append(dia)

    # for i, raw_dia in tqdm(enumerate(dataset), desc=f'preprocessing {name}'):
    #     parse_results = parse_sample(raw_dia, tokenizer, bound)
    #     if parse_results is None:
    #         continue
    #     utterances, speakers = parse_results
    #     dia = Dialogue(
    #         utterances=utterances,
    #         speakers=speakers,
    #         source_dataset_name=name,
    #         idx_within_source=i,
    #         idx=idx
    #     )
    #     idx += 1
    #     res.append(dia)
    return res


def train_test_split(data, frac=0.9, seed=0):
    """resulting sizes:
    - train: `frac`
    - test: `(1 - frac) // 2`
    - val: `(1 - frac) // 2`"""
    
    set_seet(seed)
    shuffle(data)

    n_total = len(data)
    train_size = ceil(frac * n_total)
    test_size = (n_total - train_size) // 2
    val_size = n_total - train_size - test_size

    res = {
        'train': data[:train_size],
        'test': data[train_size:train_size+test_size],
        'val': data[train_size+test_size:]
    }

    print('dataset splits sizes:')
    print(f'{n_total=}, {train_size=}, {test_size=}, {val_size=}')

    return res


def save_as_chunks(data: List[Dialogue], path, chunk_size, del_last_chunk=False):
    """saves `data` as json chunks to `save_path`"""
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    break_points = list(range(0, len(data) - chunk_size, chunk_size))
    
    if del_last_chunk:
        del break_points[-1]
    
    for i in tqdm(break_points):
        chunk_name = f'{i//chunk_size}.json'
        chunk_path = os.path.join(path, chunk_name)
        chunk = [dia.asdict() for dia in data[i:i+chunk_size]]
        json.dump(chunk, open(chunk_path, 'w'))


class DialogueDataset(Dataset):
    def __init__(self, path, fraction=1.):
        self.path = path
        
        chunk_names = [filename for filename in os.listdir(path) if filename.endswith('.json') and not filename.startswith('ru')]
        self.chunk_names = sorted(chunk_names, key=lambda x: int(x.split('.')[0]))
        
        size = math.ceil(len(self.chunk_names) * fraction)
        self.chunk_names = self.chunk_names[:size]
        
        chunk_sizes = [len(chunk) for chunk in (json.load(open(os.path.join(path, chunk_name))) for chunk_name in self.chunk_names)]
        self.chunk_beginnings = np.cumsum(chunk_sizes).tolist()

        self.n_chunks = len(self.chunk_names)
        self.len = self.chunk_beginnings[-1]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one training sample as
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
        i_chunk = bisect_right(self.chunk_beginnings, x=i)
        tmp = [0] + self.chunk_beginnings
        idx_within_chunk = i - tmp[i_chunk]
        item = json.load(open(os.path.join(self.path, self.chunk_names[i_chunk]), 'r'))[idx_within_chunk]
        return Dialogue.get_train_sample(item)