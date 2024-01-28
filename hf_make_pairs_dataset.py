import json
import numpy as np
from datasets import Dataset


# idea for training: randomly zero out some utterances from context or response during contrastive learning
# it's better not to use pairs dataset but make context-response samples on-the-fly during training

def make_pairs(dataset: Dataset, context_size, n_splits):
    for row in dataset.to_iterable_dataset():
        dia = row['content']
        split_indexes = np.random.randint(
            low=0,
            high=len(dia),
            size=n_splits
        )
        for i in split_indexes:
            start = max(0, i-context_size)
            end = i+context_size
            record = {
                'context': dia[start:i],
                'response': dia[i:end],
                'split_index': i
            }
            yield record

if __name__ == "__main__":
    from datasets import load_from_disk, DatasetDict, Dataset
    from mylib.utils.training import seed_everything
    seed_everything(0)

    input_path = 'data-2/source'
    output_path = 'data-2/train/context-response-pairs'
    
    dialog_dataset = load_from_disk(input_path)

    pairs_dataset = DatasetDict({
        split: Dataset.from_generator(
            make_pairs,
            gen_kwargs={'context_size': 3, 'n_splits': 2, 'dataset': dialog_dataset[split]},
            cache_dir=output_path
        ) for split in ['train', 'test', 'val']
    })
    
    pairs_dataset.save_to_disk(
        output_path,
        num_shards={'train': 64, 'test': 8, 'val': 8}
    )
