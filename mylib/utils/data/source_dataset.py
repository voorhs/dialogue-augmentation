import json
from random import shuffle, seed as set_seet
from math import ceil
import os
from tqdm import tqdm
from typing import List, Tuple
from .dialogue_data_type import Dialogue



def parse_sample(
        raw_sample,
        tokenizer,
        bound=None,
        user_id=0,
        system_id=1,
    ) -> Tuple[List[str], List[str]]:
    
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
    return len(tokenizer(ut)['input_ids']) <= bound


def parse_dataset(dataset, name, tokenizer, bound):
    """iterates through `dataset` and parses dialogues that satisfy 2 conditions:
    - has from 2 to 20 utterances
    - has no utterances with more than `bound` tokens
    
    If dia satisfies conditions, it is converted to `Dialogue` data type."""
    res = []
    idx = 0
    for i, raw_dia in tqdm(enumerate(dataset), desc=f'preprocessing {name}'):
        if is_empty(raw_dia) or is_too_long(raw_dia) or has_only_single_utterance(raw_dia):
            continue
        parse_results = parse_sample(raw_dia, tokenizer, bound)
        if parse_results is None:
            continue
        utterances, speakers = parse_results
        dia = Dialogue(
            utterances=utterances,
            speakers=speakers,
            source_dataset_name=name,
            idx_within_source=i,
            idx=idx
        )
        idx += 1
        res.append(dia)
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


def save_as_chunks(data: List, path, chunk_size, del_last_chunk=False):
    """saves `data` as json chunks to `save_path`"""
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    break_points = list(range(0, len(data) - chunk_size, chunk_size))
    
    if del_last_chunk:
        del break_points[-1]
    
    for i in tqdm(break_points):
        chunk_name = f'{i//chunk_size}.json'
        json.dump(data[i:i+chunk_size], open(os.path.join(path, chunk_name, 'w')))
