import os
from tqdm import tqdm

# Algorithm:
# 1. collect all json chunks into multiple hf datasets.Dataset
# 2. join Datasets
# 3. filter out those dialogues, that have no positives or negatives that are differ from original dia
#
# output samples must be of the following format:
# cur_res = {
#     'orig': original dialog
#     'pos': list of positive augmentations,
#     'neg': list of negative augmentations,
# }

def to_string(dia):
    uts = [item['utterance'] for item in dia['content']]
    res = '###'.join(uts)
    return res

def my_filter(dias, orig_joined):
    """filter out `dias` that are identical to `orig_joined`"""
    return [dia for dia in dias if to_string(dia) != orig_joined]

def make_contrastive_chunk(orig_chunk, pos_chunk, neg_chunk, allow_missing_neg):
    res = []
    for orig_dia, pos_dias, neg_dias in zip(orig_chunk, pos_chunk, neg_chunk):
        orig_dia = orig_dia[0]

        if not orig_dia['content']:
            # this case corresponds to dia that is too long (see `is_short_enough()` in filter_dataset_by_length.py)
            print('dia is too long | smth else')
            continue
        
        orig_joined = to_string(orig_dia)
        
        pos = my_filter(pos_dias, orig_joined)
        neg = my_filter(neg_dias, orig_joined)
        if not pos:
            # dia that has no positives that differ from it
            print('dia has no positives')
            continue

        if not neg:
            # dia that has no negatives that differ from it
            print('dia has no negatives')
            if not allow_missing_neg:
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
    default_aug_path = os.path.join(root_dir, 'data', 'augmented-retromae')

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path-out', dest='path_out', default='data/train/dialogue-encoder-retromae/contrastive/train')
    ap.add_argument('--orig-name', dest='orig_name', default='original')
    ap.add_argument('--aug-path', dest='aug_path', default=default_aug_path)
    ap.add_argument('--positive-names', dest='positive_names', nargs='+', default=['back-translate', 'back-translate-prune', 'back-translate-shuffle', 'prune-insert', 'insert', 'shuffle', 'prune', 'shuffle-insert'])
    ap.add_argument('--negative-names', dest='negative_names', nargs='*', default=['replace', 'replace-prune'])
    ap.add_argument('--allow-missing-neg', dest='allow_missing_neg', default=True, type=bool)
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
    #     allow_missing_neg = True
    #     chunk_size = 512
    
    # args = Args()

    # == validate input datasets ==

    positive_paths = [os.path.join(args.aug_path, x) for x in args.positive_names]
    negative_paths = [os.path.join(args.aug_path, x) for x in args.negative_names]
    original_path = os.path.join(args.aug_path, args.orig_name)

    # == generate dataset ==

    chunk_names = [filename for filename in os.listdir(positive_paths[0]) if filename.endswith('.json')]
    chunk_names = sorted(chunk_names)[:80]

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)
    
