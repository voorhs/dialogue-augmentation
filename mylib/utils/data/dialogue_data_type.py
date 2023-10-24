from pprint import pformat
from typing import List, Tuple


class Dialogue:
    def __init__(
            self,
            utterances: List[str],
            speakers: List[int],
            source_dataset_name: str,
            idx_within_source: int,
            idx: int = None,
            **fields
        ):
        """add any extra `fields` if extra info is needed to be saved"""

        if len(utterances) != len(speakers):
            raise ValueError('`utterances` and `speakers` must be the same length')
        
        self.content = [
            {'utterance': ut, 'speaker': sp} for ut, sp in zip(utterances, speakers)
        ]
        
        self.source_dataset_name = source_dataset_name
        self.idx_within_source = idx_within_source
        self.idx = idx
        
        for key, val in fields.items():
            setattr(self, key, val)
    
    def asdict(self):
        return vars(self)
    
    def __repr__(self):
        return pformat(self.asdict(), indent=2)
    
