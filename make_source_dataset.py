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
    from tqdm import tqdm
    from random import shuffle

    #! not the same as roberta, replace in future
    tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base', max_length=10000)

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
    root_dir = os.environ['ROOT_DIR_REPO']
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
        for dia in tqdm(dialogues[split], desc=f'making pairs for {split}'):
            pairs = []
            for i in range(len(dia)-1):
                pairs.append({
                    'context': dia[:i+1],
                    'target': dia[i+1]
                })
            nsp_dataset[split].extend(pairs)
        shuffle(nsp_dataset[split])
        print(split, len(nsp_dataset[split]))

    # save as chunks
    save_path = os.path.join(root_dir, 'mylib/data/train/context-response-pairs')
    for split, data in nsp_dataset.items():
        print(f'saving chunks for {split} context-response pairs')
        path = os.path.join(save_path, split)
        del_last = (split == 'test')
        save_as_chunks(data, path, chunk_size=2048, del_last_chunk=del_last)
