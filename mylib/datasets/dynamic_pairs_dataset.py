import os
from torch.utils.data import Dataset
from datasets import load_from_disk
import random


class DynamicContextResponseDataset(Dataset):
    def __init__(self, path, split, context_size):
        self.path = path
        self.split = split
        self.context_size = context_size

        self.dataset = load_from_disk(os.path.join(path, split))
    
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

        dia = self.dataset[i]['content']
        
        split_index = random.randint(a=1, b=len(dia)-2)
        start = max(0, split_index-self.context_size)
        end = split_index+self.context_size
        
        return {
            'context': dia[start:split_index],
            'target': dia[split_index:end],
        }
