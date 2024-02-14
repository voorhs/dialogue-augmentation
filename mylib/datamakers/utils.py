import os, json, yaml
from argparse import Namespace
from datetime import datetime

from datasets import Dataset
import pyarrow.parquet as pq


def dia_generator(path_in):
    json_chunks_in = sorted([filename for filename in os.listdir(path_in) if filename.endswith('.json')])

    for chunk_name in json_chunks_in:
        chunk_path = os.path.join(path_in, chunk_name)
        chunk = json.load(open(chunk_path, 'r'))
        for dia in chunk:
            dia['content'] = json.dumps(dia['content'])
            yield dia


def collect_chunks(path_in, path_out):
    """collect json chunks from `path_in` into hf datasets.Dataset and save to disk to `path_out`"""

    dataset = Dataset.from_generator(
        dia_generator,
        gen_kwargs={'path_in': path_in},
        cache_dir=False
    )
    dataset.save_to_disk(path_out)


def join(path_in, names_in, path_out):
    # read input
    tables = [pq.read_table(os.path.join(path_in, f'{name}.parquet')) for name in names_in]
    
    # join
    left = tables[0]
    for right, name in zip(tables[1:], names_in[1:]):
        left = left.join(
            right,
            keys=['id', 'source_dataset_name', 'idx_within_source'],
            join_type='inner',
            right_suffix='_'+name
        )

    # write tmp file
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    tmp = os.path.join(path_out, 'tmp.parquet')
    pq.write_table(table=left, where=tmp)

    # write output
    dataset = Dataset.from_parquet(tmp)
    dataset.save_to_disk(path_out, num_shards=64)


def _cast(row, col):
    row[col] = json.dumps(row[col])
    return row


def obj_to_str(dataset: Dataset, col):
    return dataset.map(
        _cast,
        fn_kwargs={'col': col},
        # cache_file_name=False
    )


def dump_cli_args(args: Namespace, path_out):
    now = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    yaml.dump(vars(args), open(os.path.join(path_out, f'cli_args_{now}.yml'), 'w'))
