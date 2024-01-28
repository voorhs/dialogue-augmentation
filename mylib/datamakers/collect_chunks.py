import os, json
from datasets import Dataset


def dia_generator(path_in):
    json_chunks_in = sorted([filename for filename in os.listdir(path_in) if filename.endswith('.json')])

    for chunk_name in json_chunks_in:
        chunk_path = os.path.join(path_in, chunk_name)
        chunk = json.load(open(chunk_path, 'r'))
        for dia in chunk:
            dia['content'] = json.dumps(dia['content'])
            yield dia

def main(path_in, path_out, filename):
    """collect json chunks from `path_in` into hf datasets.Dataset and save to parquet file"""

    dataset = Dataset.from_generator(
        dia_generator,
        gen_kwargs={'path_in': path_in},
        cache_dir=path_out
    )
    filename = os.path.join(path_out, f'{filename}.parquet')
    dataset.to_parquet(filename)
