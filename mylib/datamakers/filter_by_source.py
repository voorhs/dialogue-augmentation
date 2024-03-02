import os
from datasets import load_from_disk, disable_caching


def main(path_in, path_out, remove, num_shards):
    disable_caching()

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    in_dataset = load_from_disk(path_in)

    out_dataset = in_dataset.filter(lambda row: row['source_dataset_name'] not in remove)
    
    out_dataset.save_to_disk(path_out, num_shards=num_shards)
