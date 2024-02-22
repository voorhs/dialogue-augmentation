import os
from datasets import load_from_disk
from torch.utils.data import Dataset
import torch


class DomainDataset(Dataset):
    def __init__(self, path, split):
        self.path = path
        self.split = split

        self.dataset = load_from_disk(os.path.join(path, split))
    
    def __len__(self):
        return self.dataset.num_rows
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one training sample as
        ```
        {
            "type": "array",
            "items":
            {
                "type": "object",
                "properties":
                {
                    "utterance": {"type": "string"},
                    "speaker": {"type": "number"}
                }
            }
        }
        ```"""
        item = self.dataset[i]
        
        return item['content'], torch.tensor(item['services'], dtype=torch.float)
