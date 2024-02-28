import os
from functools import partial
from transformers import AutoTokenizer
from datasets import load_from_disk, disable_caching
from ..modeling.dialogue import BaselineDialogueEncoder
from ..modeling.hssa import HSSATokenizer


def is_short_enough(row, get_len, upper_bound):
    """`dia` should be shorter than 512 minus number of SEP and CLS tokens"""
    dialogues = [dia['content'] for dia in row['pos']] + [row['orig']['content']]
    return get_len(dialogues) <= upper_bound


def the_same_for_multiwoz(row, get_len, upper_bound):
    """`dia` should be shorter than 512 minus number of SEP and CLS tokens"""
    dialogues = [row['content']]
    return get_len(dialogues) <= upper_bound


def baseline_get_len(dialogues, tokenizer):
    input_ids = BaselineDialogueEncoder._tokenize(tokenizer, dialogues)['input_ids']
    return input_ids.shape[1]


def hssa_get_len(dialogues, tokenizer):
    input_ids = tokenizer(dialogues)['input_ids']
    return input_ids.shape[1]


def main(path_in, path_out, tokenizer, num_shards, upper_bound, is_hssa):
    """Copies all json chunks of a dataset from `path_in` to `path_out`.
    Each dia which is `None` or exceeding the length limit is
    replaced with `None` or dropped according to `mode`.
    
    About length limit: each dia should be shorter than 512
    minus number of SEP and CLS tokens (see `BaselineDialogueEncoder`)."""

    disable_caching()
    
    if is_hssa:
        tokenizer = HSSATokenizer.from_pretrained(tokenizer)
        get_len = partial(hssa_get_len, tokenizer=tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        get_len = partial(baseline_get_len, tokenizer=tokenizer)    

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    in_dataset = load_from_disk(path_in)
    if 'pos' in in_dataset[0].keys():
        is_accepted = is_short_enough
    else:
        is_accepted = the_same_for_multiwoz

    out_dataset = in_dataset.filter(
        is_accepted,
        fn_kwargs=dict(
            get_len=get_len,
            upper_bound=upper_bound
        )
    )
    
    out_dataset.save_to_disk(path_out, num_shards=num_shards)
