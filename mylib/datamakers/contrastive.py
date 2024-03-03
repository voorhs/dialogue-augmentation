import os, shutil, json
from datasets import load_from_disk, disable_caching
from mylib.datamakers.utils import join, collect_chunks, obj_to_str


def source_to_parquet():
    source_dataset = 'data/source/train'
    source_parquet = 'data/augmented/source.parquet'
    source = load_from_disk(source_dataset)
    source = obj_to_str(source, col='content')
    source.to_parquet(source_parquet)
    return source_parquet


def remove_parquets(paths):
    for path in paths:
        os.remove(path)


def remove_datasets(paths):
    for path in paths:
        shutil.rmtree(path)


def _gather(row: dict, content_keys):
    pos = []
    for k in content_keys:
        if k == 'content':
            continue

        if row[k] is None:
            continue

        pos.append({
            'content': json.loads(row[k]),
            'augmentation': k[8:],
        })
    row['pos'] = pos
    row['orig'] = {
        'content': json.loads(row['content']),
        'augmentation': None
    }
    return row


def gather_positives_to_list(dataset):
    row = dataset[0]
    content_keys = [k for k in row.keys() if k.startswith('content')]
    return dataset.map(
        _gather,
        fn_kwargs=dict(content_keys=content_keys),
        # cache_file_name='data/cache-2',
        remove_columns=content_keys
    )


def main(path, names_in, name_out):
    disable_caching()

    # collect chunks of augmented datasets
    for name in names_in:
        collect_chunks(
            path_in=os.path.join(path, name),
            path_out=os.path.join(path, f'{name}-collected')
        )

    # convert to parquet all datasets
    source_parquet = source_to_parquet()
    parquet_paths = [os.path.join(path, f'{n}.parquet') for n in names_in]
    dataset_paths = [os.path.join(path, f'{n}-collected') for n in names_in]

    for p, d in zip(parquet_paths, dataset_paths):
        load_from_disk(d).to_parquet(p)

    # === finally ===
    path_out = os.path.join('data', name_out)
    path_out_tmp = path_out + '-tmp'
    join(
        path_in=path,
        names_in=['source']+names_in,
        path_out=path_out_tmp,
    )
    
    # convert to desired format
    dataset = load_from_disk(path_out_tmp)
    dataset = gather_positives_to_list(dataset)
    dataset.save_to_disk(path_out)

    # clear tmp files
    parquet_paths += [source_parquet]
    remove_parquets(parquet_paths)
    remove_datasets(dataset_paths+[path_out_tmp])
