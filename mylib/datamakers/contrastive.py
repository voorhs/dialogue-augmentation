import os
from datasets import load_from_disk
from mylib.datamakers.utils import join, collect_chunks, obj_to_str


def source_to_parquet():
    source_dataset = 'data-2/source/train'
    source_parquet = 'data-2/augmented/source.parquet'
    source = load_from_disk(source_dataset)
    source = obj_to_str(source, col='content')
    source.to_parquet(source_parquet)


def main(path, names_in, name_out):
    # collect chunks of augmented datasets
    for name in names_in:
        collect_chunks(
            path_in=os.path.join(path, name),
            path_out=os.path.join(path, f'{name}-collected')
        )

    # convert to parquet all datasets
    source_to_parquet()
    parquet_paths = [os.path.join(path, f'{n}.parquet') for n in names_in]
    dataset_paths = [os.path.join(path, f'{n}-collected') for n in names_in]

    for p, d in zip(parquet_paths, dataset_paths):
        load_from_disk(d).to_parquet(p)

    # === finally ===
    join(
        path_in=path,
        names_in=['source']+names_in,
        path_out=os.path.join('data-2', name_out),
    )
