import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/alekseev_ilya/dialogue-augmentation/nup')
from models import SimpleDialogueEncoder


def filter_dataset(path_in, path_out, tokenizer='roberta-base'):
    """copies all json chunks of a dataset from `path_in` to `path_out`. Each dia exceeding the length limit is replaced with None."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def is_short_enough(dia, upper_bound=512):
        """dia should be shorter than 512 minus number of SEP and CLS tokens"""
        input_ids = SimpleDialogueEncoder._tokenize(tokenizer, [dia])['input_ids'][0]
        return len(input_ids) <= upper_bound

        
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    chunk_names = [filename for filename in os.listdir(path_in) if filename.endswith('.json') and not filename.startswith('ru')]
    chunk_names = sorted(chunk_names, key=lambda x: int(x.split('.')[0]))
    
    for chunk_name in tqdm(chunk_names):
        chunk_path_in = os.path.join(path_in, chunk_name)
        chunk = json.load(open(chunk_path_in, 'r'))
        
        filtered_chunk = []
        for i, dia in enumerate(chunk):
            res = dia
            if dia is not None and not is_short_enough(dia):
                res = None
                print(f'rejected dia #{i}')
            filtered_chunk.append(res)
        
        chunk_path_out = os.path.join(path_out, chunk_name)
        json.dump(filtered_chunk, open(chunk_path_out, 'w'))


if __name__ == "__main__":
    dir_path_in = '/home/alekseev_ilya/dialogue-augmentation/augmented/'
    input_names = [
        'back-translate',
        'back-translate-cut',
        'back-translate-shuffle',
        'cut-insert',
        'insert',
        'pairwise-shuffler',
        'pairwise-cutter',
        'replace',
        'replace-cut',
        'shuffle-insert'
    ]
    dir_path_out = '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented'
    
    for name in input_names:
        path_in = os.path.join(dir_path_in, name)
        path_out = os.path.join(dir_path_out, name)

        filter_dataset(path_in, path_out)
