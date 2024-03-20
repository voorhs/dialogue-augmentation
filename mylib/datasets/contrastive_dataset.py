from torch.utils.data import Dataset
from datasets import load_from_disk
from random import choice, randint


class ContrastiveDataset(Dataset):
    def __init__(self, path):
        self.path = path

        self.dataset = load_from_disk(path)
        self.augs = [precomputed, halves, response_selection, span_matching]

    def __len__(self):
        return self.dataset.num_rows
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one training sample as
        {
            "orig": orig dia's content,
            "pos": one of augmentations' content,
        }
        """
        sample = self.dataset[i]
        aug = choice(self.augs)
        return aug(sample)


def get_split_index(dia):
    if len(dia) == 2:
        return 1
    else:
        return randint(a=1, b=len(dia)-2)


def halves(sample):
    dia = sample['orig']['content']
    
    split_index = get_split_index(dia)

    return dia[:split_index], dia[split_index:]


def precomputed(sample):
    orig = sample['orig']['content']
    
    positive = choice(sample['pos'])['content']

    return orig, positive


def response_selection(sample):
    dia = sample['orig']['content']

    split_index = get_split_index(dia)

    return dia[max(0,split_index-3):split_index], [dia[split_index]]


def span_matching(sample):
    dia = sample['orig']['content']

    if len(dia) <= 3:
        return halves(sample)
    
    start = randint(a=1, b=len(dia)-2)
    end = randint(a=start+1, b=len(dia)-1)

    span = dia[start:end]
    negate_span = dia[:start] + dia[end:]
    
    return span, negate_span
