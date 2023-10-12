from torch.utils.data import Dataset
import math
import os
import json


class DialogueDataset(Dataset):
    chunk_size = 512
    def __init__(self, path, n_chunks=80):
        self.path = path
        self.chunk_fnames = sorted([filename for filename in os.listdir(path) if filename.endswith('.json')])[:n_chunks]
        self.n_chunks = len(self.chunk_fnames)
        self.len = self.n_chunks * self.chunk_size
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one dialogue, represented with an object of the following schema:
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
        i_chunk = math.floor(i / self.chunk_size)
        idx_within_chunk = i % self.chunk_size
        item = json.load(open(os.path.join(self.path, self.chunk_fnames[i_chunk]), 'r'))[idx_within_chunk]
        return item