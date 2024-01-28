from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset
from mylib.utils.training import seed_everything


def get_record_generator(split):
    dataset = load_dataset('multi_woz_v22', split=split)
    for i, sample in enumerate(dataset):
        uts = sample['turns']['utterance']
        sps = sample['turns']['speaker']
        yield dict(
            content=[{'utterance': ut, 'speaker': sp} for ut, sp in zip(uts, sps)],
            source_dataset_name=f'multiwoz/{split}',
            idx_within_source=i,
            services=sample['services']
        )
    

def main(output_path, seed):
    seed_everything(seed)
    
    dataset = DatasetDict({
        split: Dataset.from_generator(
            get_record_generator,
            gen_kwargs=dict(split=split),
            cache_dir=output_path
        ) for split in ['train', 'validation', 'test']
    }) 
    

    dataset.save_to_disk(output_path, num_shards={'train': 4, 'test': 1, 'val': 1})
