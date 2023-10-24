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


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from collections import defaultdict
    from mylib.utils.data.source_dataset import parse_dataset, train_test_split, save_as_chunks
    from mylib.utils.data.pairs_dataset import make_pairs
    from mylib.utils.training import seed_everything

    seed_everything(0)

    # supress warnings about long sequences
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    #! not the same as roberta, replace in future
    tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base')

    # load datasets from hugging face, parse, filter and merge into single list
    merged_dataset = []
    for name in names:
        dataset = load_dataset('Salesforce/dialogstudio', name)['train']['log']
        parsed_dataset = parse_dataset(dataset, name, tokenizer, upper_bounds[name])
        merged_dataset.extend(parsed_dataset)


    # shuffle and define splits
    dialogues = train_test_split(merged_dataset)


    # save splits to file system as json chunks ('mylib/data/train/source')
    import os
    root_dir = os.environ['REPO_DIR']
    save_path = os.path.join(root_dir, 'mylib/data/train/source')

    for split, data in dialogues.items():
        print(f'saving chunks for {split} dialogues')
        path = os.path.join(save_path, split)
        save_as_chunks(data, path, chunk_size=512)

    # === context-response pairs dataset ===

    # make pairs
    save_path = os.path.join(root_dir, 'mylib/data/train/context-response-pairs')
    nsp_dataset = defaultdict(list)
    for split in ['train', 'test', 'val']:
        nsp_dataset[split] = make_pairs(dialogues[split])
        print(split, len(nsp_dataset[split]))

    # save as chunks
    save_path = os.path.join(root_dir, 'mylib/data/train/context-response-pairs')
    for split, data in nsp_dataset.items():
        print(f'saving chunks for {split} context-response pairs')
        path = os.path.join(save_path, split)
        del_last = (split == 'test')
        save_as_chunks(data, path, chunk_size=2048, del_last_chunk=del_last)
