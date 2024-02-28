import os
from torch.utils.data import Dataset
from datasets import load_from_disk
import random


class HalvesDataset(Dataset):
    def __init__(self, path):
        self.path = path

        self.dataset = load_from_disk(path)
    
    def __len__(self):
        return self.dataset.num_rows
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one dialogue, represented with an object of the following schema:
        ```
        {
            "type": "object",
            "properties":
            {
                "context":
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
                },
                "target": same as "context"
            }
        }
        ```"""

        dia = self.dataset[i]['orig']['content']
        
        if len(dia) == 2:
            split_index = 1
        else:
            split_index = random.randint(a=1, b=len(dia)-2)

        return {
            'orig': {'content': dia[:split_index]},
            'pos': [{'content': dia[split_index:]}],
        }
