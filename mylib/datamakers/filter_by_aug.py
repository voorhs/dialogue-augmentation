import os
from datasets import load_from_disk, disable_caching


def main(path_in, path_out, remove, num_shards):
    disable_caching()

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    in_dataset = load_from_disk(path_in)

    out_dataset = in_dataset.map(
        lambda pos: {'pos': [aug for aug in pos if aug['augmentation'] not in remove]},
        input_columns='pos',
    )
    
    out_dataset.save_to_disk(path_out, num_shards=num_shards)
