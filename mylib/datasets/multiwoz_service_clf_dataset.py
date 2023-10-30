import os
import json
from bisect import bisect_right
import torch
from .source_dataset import DialogueDataset

class MultiWOZServiceClfDataset(DialogueDataset):
    services = [
            'attraction', 'bus', 'hospital',
            'hotel', 'restaurant', 'taxi', 'train'
        ]
    def __getitem__(self, i):
        i_chunk = bisect_right(self.chunk_beginnings, x=i)
        tmp = [0] + self.chunk_beginnings
        idx_within_chunk = i - tmp[i_chunk]
        item = json.load(open(os.path.join(self.path, self.chunk_names[i_chunk]), 'r'))[idx_within_chunk]
        
        dia = item['content']
        services = item['services']
        
        target = torch.tensor([float(serv in services) for serv in self.services])    # multi one hot
        
        return dia, target
    
