import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path-in', dest='path_in', required=True)
    ap.add_argument('--path-out', dest='path_out', required=False)
    args = ap.parse_args()
    

    tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base', max_length=10000)
    

    def is_short_enough(dia, upper_bound=512):
        """dia should be shorter than 512 minus number of SEP and CLS tokens"""
        uts = [item['utterance'] for item in dia]
        joined = ' '.join(uts)
        tokens = tokenizer(joined)['input_ids']
        upper_bound -= len(uts) + 1
        return len(tokens) <= upper_bound

        
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)
    chunk_names = [filename for filename in os.listdir(args.path_in) if filename.endswith('.json') and not filename.startswith('ru')]
    chunk_names = sorted(chunk_names, key=lambda x: int(x.split('.')[0]))
    
    for chunk_name in tqdm(chunk_names):
        chunk_path_in = os.path.join(args.path_in, chunk_name)
        chunk = json.load(open(chunk_path_in, 'r'))
        
        filtered_chunk = []
        for i, dia in enumerate(chunk):
            res = dia
            if dia is not None and not is_short_enough(dia):
                res = None
                print(f'rejected dia #{i}')
            filtered_chunk.append(res)
        
        chunk_path_out = os.path.join(args.path_out, chunk_name)
        json.dump(filtered_chunk, open(chunk_path_out, 'w'))