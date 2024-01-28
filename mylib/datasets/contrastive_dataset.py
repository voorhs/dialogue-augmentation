from torch.utils.data import Dataset
from datasets import load_from_disk


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
            'orig': dia,
            'pos': list of dias,
        }
        where each dia is represented with an object of the following schema:
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
        return self.dataset[i]
