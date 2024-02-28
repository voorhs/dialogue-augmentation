import os, json

from datasets import Dataset
import pyarrow.parquet as pq


def dia_generator(path_in, as_string=True):
    json_chunks_in = sorted([filename for filename in os.listdir(path_in) if filename.endswith('.json')])

    for chunk_name in json_chunks_in:
        chunk_path = os.path.join(path_in, chunk_name)
        chunk = json.load(open(chunk_path, 'r'))
        for dia in chunk:
            content = dia['content']
            if content is None or content[0] is None:
                # some of augmentations were impossible, so `None` or `[None, -inf]` is stored
                continue
            if as_string:
                dia['content'] = json.dumps(dia['content'])
            yield dia


def collect_chunks(path_in, path_out, as_string=True):
    """collect json chunks from `path_in` into hf datasets.Dataset and save to disk to `path_out`"""

    dataset = Dataset.from_generator(
        dia_generator,
        gen_kwargs={'path_in': path_in, 'as_string': as_string},
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
