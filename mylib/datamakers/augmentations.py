import json, os

from tqdm.auto import tqdm
from datasets import load_from_disk

from mylib.augmentations import *


def get_augmenter(method):
    if method == 'back-translate':
        augmenter = BackTranslator(language='ru', device='cuda')
    elif method == 'insert':
        augmenter = Inserter(
            fraction=0.5,
            score_threshold=0.005,
            k=5,
            mask_utterance_level=True,
            fill_utterance_level=2,
            model='FacebookAI/roberta-base',
            device='cuda'
        )
    elif method == 'replace':
        augmenter = Replacer(
            k=3,
            fill_utterance_level=2,
            model='FacebookAI/roberta-base',
            device='cuda'
        )
    elif method == 'prune':
        augmenter = Pruner(device='cuda')
    elif method == 'shuffle':
        augmenter = Shuffler(device='cuda')
    
    return augmenter


def get_path_out(dir_name):
    path_out = os.path.join(os.getcwd(), 'data', 'augmented', dir_name)
    
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    
    return path_out


def get_batch_loader(batch_size, method_name):
    dataset = load_from_disk('data/source/train')

    batch_loader = dataset.iter(batch_size=batch_size, drop_last_batch=False)
    return tqdm(
        enumerate(batch_loader),
        desc=method_name,
        total=dataset.num_rows // batch_size
    )


def convert(batch, batch_size):
    """covert from one batched format (dict of lists) to another batched format (list of dicts)"""
    return [{key: batch[key][i] for key in batch.keys()} for i in range(batch_size)]


def main(method_name, dir_name, batch_size, skip_existing):
    augmenter = get_augmenter(method_name)
    path_out = get_path_out(dir_name)
    batch_loader = get_batch_loader(batch_size, method_name)
    json_chunks_out = sorted([filename for filename in os.listdir(path_out) if filename.endswith('.json')])
    print('result will be saved to', path_out)

    for i_batch, batch in batch_loader:
        # check if chunk already exists in directory
        chunk_name = f'{i_batch:05d}.json'
        if chunk_name in json_chunks_out and skip_existing:
            print(f'skipping {chunk_name}')
            continue
        
        # augment
        batch = convert(batch, batch_size)
        orig_contents = [dia['content'] for dia in batch]
        aug_contents = augmenter(orig_contents)

        # replace original content with augmented one
        for dia, content in zip(batch, aug_contents):
            dia['content'] = content

        json.dump(batch, open(os.path.join(path_out, chunk_name), 'w'))
