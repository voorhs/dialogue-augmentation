names = [
    'MS-DC',
    'MetaLWOZ',
    'MULTIWOZ2_2',
    'SGD',
    'SimJointGEN',
    'KETOD',
    'FRAMES',
    'Disambiguation',
    'ABCD',
    'AirDialogue',
    'BiTOD',
    'Taskmaster1'
]

upper_bound = 96

upper_bounds = {
    'MS-DC': min(upper_bound, 250),
    'MetaLWOZ': min(upper_bound, 100),
    'MULTIWOZ2_2': min(upper_bound, 75),
    'SGD': None,
    'SimJointGEN': upper_bound,
    'KETOD': upper_bound,
    'FRAMES': upper_bound,
    'Disambiguation': min(upper_bound, 60),
    'ABCD': upper_bound,
    'AirDialogue': upper_bound,
    'BiTOD': upper_bound,
    'Taskmaster1': min(upper_bound, 200),
}


from random import shuffle, seed as set_seet
from math import ceil
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from typing import Tuple, List
from tqdm import tqdm
from mylib.utils.data import Dialogue, ContextResponsePair
import json
import os
import pyarrow as pa
from datasets import load_dataset


def preprocess_dialogue(
        raw_sample,
        tokenizer,
        bound=None,
        user_id=0,
        system_id=1,
    ):
    """convert single dialogue (in DialogStudio format) to list of utterances and list of corresponding speaker ids"""
    
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


def get_record_iterator(name, tokenizer, bound):
    dataset = load_dataset('Salesforce/dialogstudio', name)['train']['log']
    for i, raw_dia in tqdm(enumerate(dataset), desc=f'parsing {name}'):
        dia = preprocess_dialogue(raw_dia, tokenizer, bound)
        if dia is None:
            continue
        utterances, speakers = dia
        dia = Dialogue(
            utterances=utterances,
            speakers=speakers,
            source_dataset_name=name,
            idx_within_source=i,
            # idx=None
        )
        dct = dia.asdict()
        dct['content'] = json.dumps(dct['content'])
        yield pa.RecordBatch.from_pylist([dct])


def train_test_split(data: List[Dialogue], frac=0.9, seed=0):
    """resulting sizes:
    - train: `frac`
    - test: `(1 - frac) // 2`
    - val: `(1 - frac) // 2`"""
    
    set_seet(seed)
    shuffle(data)

    # assign indices after shuffling the dataset
    for i, dia in enumerate(data):
        dia.idx = i

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


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from collections import defaultdict
    from mylib.utils.training import seed_everything
    import itertools as it
    import pyarrow.dataset as ds

    seed_everything(0)

    # supress warnings about long sequences
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    #! not the same as roberta, replace in future
    tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base')

    # load datasets from hugging face, parse, filter and merge into single list
    record_iterator_list = []
    for dataset_name in names:
        iterator = get_record_iterator(dataset_name, tokenizer, upper_bounds[dataset_name])
        record_iterator_list.append(iterator)
    
    chained_iterator = it.chain.from_iterable(record_iterator_list)

    ds.write_dataset(
        data=chained_iterator,
        base_dir='data-2/source',
        max_rows_per_file=512,
        max_rows_per_group=512,
        schema=pa.schema([
            ('content', pa.string()),
            ('idx_within_source', pa.int32()),
            ('source_dataset_name', pa.string()),
        ]),
        format='parquet',
        use_threads=False
    )


    # # shuffle and define splits
    # dialogues = train_test_split(merged_dataset)


    # # save to file system
    # import os
    # root_dir = os.environ['ROOT_DIR']
    # save_path = os.path.join(root_dir, 'data', 'source')

    # for split, data in dialogues.items():
    #     print(f'saving chunks for {split} dialogues')
    #     path = os.path.join(save_path, split)
    #     save_as_chunks(data, path, chunk_size=512)

    # # === context-response pairs dataset ===

    # # make pairs
    # save_path = os.path.join(root_dir, 'data', 'train', 'context-response-pairs')
    # nsp_dataset = defaultdict(list)
    # for split in ['train', 'test', 'val']:
    #     nsp_dataset[split] = make_pairs(dialogues[split])
    #     print(split, len(nsp_dataset[split]))

    # # save as chunks
    # save_path = os.path.join(root_dir, 'data', 'train', 'context-response-pairs')
    # for split, data in nsp_dataset.items():
    #     print(f'saving chunks for {split} context-response pairs')
    #     path = os.path.join(save_path, split)
    #     del_last = (split == 'test')
    #     save_as_chunks(data, path, chunk_size=2048, del_last_chunk=del_last)
