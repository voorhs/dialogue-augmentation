import os
from transformers import AutoTokenizer
from datasets import load_from_disk
from ..modeling.dialogue import BaselineDialogueEncoder


def is_short_enough(row, tokenizer, upper_bound):
    """`dia` should be shorter than 512 minus number of SEP and CLS tokens"""
    dialogues = [dia['content'] for dia in row['pos']] + [row['orig']['content']]
    input_ids = BaselineDialogueEncoder._tokenize(tokenizer, dialogues)['input_ids']
    return input_ids.shape[1] <= upper_bound


def main(path_in, path_out, tokenizer, upper_bound=512):
    """Copies all json chunks of a dataset from `path_in` to `path_out`.
    Each dia which is `None` or exceeding the length limit is
    replaced with `None` or dropped according to `mode`.
    
    About length limit: each dia should be shorter than 512
    minus number of SEP and CLS tokens (see `BaselineDialogueEncoder`)."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
 
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    in_dataset = load_from_disk(path_in)
    out_dataset = in_dataset.filter(
        is_short_enough,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            upper_bound=upper_bound
        )
    )
    
    out_dataset.save_to_disk(path_out, num_shards=64)
