import os
from torch.utils.data import Dataset
from datasets import load_from_disk


class DialogueDataset(Dataset):
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
        return self.dataset[i]['content']
