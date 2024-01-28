names = [
    'MS-DC',
    'MetaLWOZ',
    'MULTIWOZ2_2',
    'SGD',
    'SimJointGEN',
    'KETOD',
    'FRAMES',
    'Disambiguation',
    'ABCD',
    'AirDialogue',
    'BiTOD',
    'Taskmaster1'
]

upper_bound = 96

upper_bounds = {
    'MS-DC': min(upper_bound, 250),
    'MetaLWOZ': min(upper_bound, 100),
    'MULTIWOZ2_2': min(upper_bound, 75),
    'SGD': None,
    'SimJointGEN': upper_bound,
    'KETOD': upper_bound,
    'FRAMES': upper_bound,
    'Disambiguation': min(upper_bound, 60),
    'ABCD': upper_bound,
    'AirDialogue': upper_bound,
    'BiTOD': upper_bound,
    'Taskmaster1': min(upper_bound, 200),
}

from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset


def preprocess_dialogue(
        raw_sample,
        tokenizer,
        bound=None,
        user_id=0,
        system_id=1,
    ):
    """convert single dialogue (in DialogStudio format) to list of utterances and list of corresponding speaker ids"""
    
    if is_empty(raw_sample) or is_too_long(raw_sample) or has_only_single_utterance(raw_sample):
        return

    utterances = []
    speakers = []
    
    for turn in raw_sample:
        for sp, item in zip([user_id, system_id], ['user utterance', 'system response']):
            ut = turn[item]
            if ut == '':
                continue
            utterances.append(ut)
            speakers.append(sp)
    
    if bound is not None and any(is_above_bound(ut, tokenizer, bound) for ut in utterances):
        # if there're any utterances with exceeding length
        return
    
    return utterances, speakers


def is_empty(dia):
    return len(dia) == 0


def is_too_long(dia):
    return len(dia) > 10


def has_only_single_utterance(dia):
    return len(dia) == 1 and (dia[0]['user utterance'] == '' or dia[0]['system response'] == '')


def is_above_bound(ut, tokenizer, bound):
    return len(tokenizer(ut)['input_ids']) > bound


class Counter:
    def __init__(self):
        self.counter = 0
    
    def __call__(self):
        self.counter += 1
        return self.counter

counter = Counter()


def get_record_generator(name, tokenizer, bound):
    dataset = load_dataset('Salesforce/dialogstudio', name)['train']['log']
    for i, raw_dia in tqdm(enumerate(dataset), desc=f'parsing {name}'):
        dia = preprocess_dialogue(raw_dia, tokenizer, bound)
        if dia is None:
            continue
        utterances, speakers = dia
        yield {
            'content': [{'utterance': ut, 'speaker': sp} for ut, sp in zip(utterances, speakers)],
            'source_dataset_name': name,
            'idx_within_source': i,
            'id': counter()
        }


def train_test_split(dataset: Dataset):
    train_test = dataset.train_test_split(test_size=.1, shuffle=True, seed=0)
    test_val = train_test['test'].train_test_split(test_size=.5, shuffle=False)
    test_val['val'] = test_val['train']
    res_dataset = DatasetDict({
        'train': train_test['train'],
        'test': test_val['test'],
        'val': test_val['val']
    })
    return res_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from mylib.utils.training import seed_everything
    from datasets import Dataset

    seed_everything(0)

    output_path = 'data-2/source'

    # supress warnings about long sequences
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    #! not the same as roberta, replace in future
    tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base')

    # load datasets from hugging face, parse, filter and merge into single list
    def chained_generator():
        for dataset_name in names:
            generator = get_record_generator(dataset_name, tokenizer, upper_bounds[dataset_name])
            yield from generator

    # main line of code that creates dataset
    dialog_dataset = Dataset.from_generator(chained_generator, cache_dir=output_path)
    
    # make splits and save to disk
    dialog_dataset = train_test_split(dialog_dataset)
    dialog_dataset.save_to_disk(output_path, num_shards={'train': 64, 'test': 8, 'val': 8})
