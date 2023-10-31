# negative_names = [
#     'replace',
#     'replace-prune',
# ]

# positive_names = [
#     'back-translate',
#     'back-translate-prune',
#     'back-translate-shuffle',
#     'prune-insert',
#     'insert',
#     'shuffle',
#     'prune',
#     'shuffle-insert',
# ]

import json
import os
from tqdm import tqdm

def validate(paths, n_chunks):
    """each of the provided data sets for positive, negative
    and original dialogues - each of them must
    be the same in all fields except content

    function `validate` checks only chunk sizes and chunk names
    """
    all_chunk_names = []
    for pos_path in tqdm(paths, desc='validating provided paths'):
        chunk_names = [filename for filename in os.listdir(pos_path) if filename.endswith('.json')]
        if len(chunk_names) != n_chunks:
            print('wrong number of chunks:', pos_path)
        for chunk_name in chunk_names:
            chunk_path = os.path.join(pos_path, chunk_name)
            chunk = json.load(open(chunk_path, 'r'))
            if len(chunk) != 512:
                print('wrong chunk size:', chunk_name, pos_path)
        all_chunk_names.append(chunk_names)

    for chunk_names in all_chunk_names[1:]:
        if any(name1 != name2 for name1, name2 in zip(all_chunk_names[0], chunk_names)):
            print('chunk names must match')

#! add info aggregation
def read_chunk(paths, chunk_name):
    """read `chunk_name` from all `paths`
    and return as list of tuples (dia, dia...) for each dia in paths.
    In some sense, it works as `np.stack(paths, axis=1)`."""
    res = []
    for path in paths:
        chunk_path = os.path.join(path, chunk_name)
        chunk = json.load(open(chunk_path, 'r'))
        res.append(chunk)
    return [[dia for dia in dias if dia['content'] is not None] for dias in zip(*res)]

def join(dia):
    uts = [item['utterance'] for item in dia['content']]
    res = '###'.join(uts)
    return res

def my_filter(dias, orig_joined):
    """filter out `dias` that are identical to `orig_joined`"""
    return [dia for dia in dias if join(dia) != orig_joined]

def make_contrastive_chunk(orig_chunk, pos_chunk, neg_chunk, allow_absent_neg):
    res = []
    for orig_dia, pos_dias, neg_dias in zip(orig_chunk, pos_chunk, neg_chunk):
        if not orig_dia:
            print('its a multiwoz')
            continue
        orig_dia = orig_dia[0]

        if not orig_dia['content']:
            # this case corresponds to dia that is too long (see `is_short_enough()` in filter_dataset_by_length.py)
            print('dia is too long | smth else')
            continue
        
        orig_joined = join(orig_dia)
        
        pos = my_filter(pos_dias, orig_joined)
        neg = my_filter(neg_dias, orig_joined)
        if not pos:
            # dia that has no positives that differ from it
            print('dia has no positives')
            continue

        if not neg:
            # dia that has no negatives that differ from it
            print('dia has no negatives')
            if not allow_absent_neg:
                continue

        cur_res = {
            'orig': orig_dia,
            'pos': pos,
            'neg': neg,
        }
        res.append(cur_res)
    return res

if __name__ == "__main__":
    import os
    root_dir = os.environ['ROOT_DIR']
    default_aug_path = os.path.join(root_dir, 'data', 'augmented')

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path-out', dest='path_out', default='data/train/dialogue-encoder-bert-base-cased/contrastive/train')
    ap.add_argument('--orig-name', dest='orig_name', default='original')
    ap.add_argument('--aug-path', dest='aug_path', default=default_aug_path)
    ap.add_argument('--positive-names', dest='positive_names', nargs='+', default=['back-translate', 'back-translate-prune', 'back-translate-shuffle', 'prune-insert', 'insert', 'shuffle', 'prune', 'shuffle-insert'])
    ap.add_argument('--negative-names', dest='negative_names', nargs='*', default=['replace', 'replace-prune'])
    ap.add_argument('--allow-absent-neg', dest='allow_absent_neg', default=True, type=bool)
    ap.add_argument('--chunk-size', dest='chunk_size', default=512, type=int)
    args = ap.parse_args()

    # from dataclasses import dataclass

    # @dataclass
    # class Args:
    #     path_out = 'data/train/dialogue-encoder/contrastive/train'
    #     orig_name = 'truncated-bert-base-cased'
    #     aug_path = 'data/augmented'
    #     positive_names = ['back-translate', 'back-translate-prune', 'back-translate-shuffle', 'prune-insert', 'insert', 'shuffle', 'prune', 'shuffle-insert']
    #     negative_names = ['replace', 'replace-prune']
    #     allow_absent_neg = True
    #     chunk_size = 512
    
    # args = Args()

    # == validate input datasets ==

    positive_paths = [os.path.join(args.aug_path, x) for x in args.positive_names]
    negative_paths = [os.path.join(args.aug_path, x) for x in args.negative_names]
    original_path = os.path.join(args.aug_path, args.orig_name)

    # validate(positive_paths + negative_paths + [args.orig_path], args.chunk_size)

    #! remove [:80]
    # == generate dataset ==

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)
    
    chunk_names = [filename for filename in os.listdir(positive_paths[0]) if filename.endswith('.json')]
    chunk_names = sorted(chunk_names)[:80]
    
    for chunk_name in tqdm(chunk_names, desc='generating dataset'):
        orig_chunk = read_chunk([original_path], chunk_name)
        pos_chunk = read_chunk(positive_paths, chunk_name)
        neg_chunk = read_chunk(negative_paths, chunk_name)
        
        chunk = make_contrastive_chunk(orig_chunk, pos_chunk, neg_chunk, args.allow_absent_neg)
        
        chunk_path = os.path.join(args.path_out, chunk_name)
        json.dump(chunk, open(chunk_path, 'w'))
