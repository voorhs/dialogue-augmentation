from datasets import load_from_disk
from transformers import AutoTokenizer
from mylib.modeling.dialogue import BaselineDialogueEncoder


def get_aug_indicators(pos, target):
    augs = [aug['augmentation'] for aug in pos]
    return {t: int(t in augs) for t in target}


def n_tokens(pos, tokenizers, target):
    res = {}
    
    for name, tok in tokenizers.items():
        for t in target:
            res[f'n-toks-{name}-{t}'] = None

    for name, tok in tokenizers.items():
        for aug in pos:
            input_ids = BaselineDialogueEncoder._tokenize(tok, [aug['content']])['input_ids']
            res[f'n-toks-{name}-{aug["augmentation"]}'] = input_ids.shape[1]
    return res


def n_utterances(pos):
    res = {}
    for aug in pos:
        res[f'n-uts-{aug["augmentation"]}'] = len(aug['content'])
    return res


def make_report(pack):
    dataset = load_from_disk(f'data/train/{pack}')

    if pack == 'trivial':
        target = ['back-translate', 'insert', 'replace']
    elif pack == 'advanced':
        target = ['back-translate', 'insert', 'replace', 'prune', 'shuffle']
    elif pack == 'crazy':
        target = ['back-translate', 'insert', 'replace', 'prune', 'shuffle', 'back-translate-prune', 'prune-insert', 'prune-replace', 'shuffle-insert', 'shuffle-replace']
    else:
        raise ValueError('unknown pack')

    dataset = dataset.map(
        function=get_aug_indicators,
        fn_kwargs=dict(target=target),
        input_columns='pos',
        desc='getting aug indicators'
    )
    
    dataset = dataset.map(
        function=n_tokens,
        fn_kwargs=dict(
            tokenizers={
                'bert': AutoTokenizer.from_pretrained('google-bert/bert-base-uncased'),
                'roberta': AutoTokenizer.from_pretrained('FacebookAI/roberta-base'),
                'retromae': AutoTokenizer.from_pretrained('Shitao/RetroMAE')
            },
            target=target
        ),
        input_columns='pos',
        desc='counting dia tokens'
    )

    dataset = dataset.map(function=n_utterances, input_columns='pos', desc='counting utterances')
    dataset = dataset.remove_columns(['source_dataset_name', 'idx_within_source', 'id', 'pos', 'orig'])
    dataset.save_to_disk(f'data/reports/{pack}')


if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('--pack', dest='pack', required=True, choices=['trivial', 'advanced', 'crazy'])
    args = ap.parse_args()

    make_report(args.pack)
