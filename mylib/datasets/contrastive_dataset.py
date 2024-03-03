from torch.utils.data import Dataset
from datasets import load_from_disk
from random import choice


class ContrastiveDataset(Dataset):
    def __init__(self, path):
        self.path = path

        self.dataset = load_from_disk(path)

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
        sample =  self.dataset[i]
        orig = sample['orig']['content']
        
        positive = choice(sample['pos'])['content']

        return orig, positive
