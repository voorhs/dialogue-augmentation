import os
import pyarrow.parquet as pq
from datasets import Dataset


def main(files_in, names, path_out, name_out):
    tables = [pq.read_table(file) for file in files_in]
    left = tables[0]
    for right, name in zip(tables[1:], names[1:]):
        left = left.join(right, keys=['id', 'source_dataset_name', 'idx_within_source'], join_type='inner', right_suffix=name)
    
    file_out = os.path.join(path_out, 'tmp.parquet')
    pq.write_table(table=left, where=file_out)

    dataset = Dataset.from_parquet(file_out)
    path_out = os.path.join(path_out, name_out)
    dataset.save_to_disk(path_out, num_shards=64)
