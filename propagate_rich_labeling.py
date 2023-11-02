if __name__ == "__main__":
    import json
    import os
    from tqdm import tqdm

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--without', dest='without', required=True)
    ap.add_argument('--with', dest='with_', required=True)
    args = ap.parse_args()

    without_labeling = args.without
    with_labeling = args.with_

    chunk_names = [fname for fname in os.listdir(without_labeling) if fname.endswith('.json')]

    for chunk_name in tqdm(chunk_names):
        path_without = os.path.join(without_labeling, chunk_name)
        chunk_without_labeling = json.load(open(path_without, 'r'))
        if isinstance(chunk_without_labeling[0], dict):
            print('already has labeling')
            continue
        print('making')
        path_with = os.path.join(with_labeling, chunk_name)
        chunk_with_labeling = json.load(open(path_with, 'r'))
        for dia_with, dia_without in zip(chunk_with_labeling, chunk_without_labeling):
            dia_with['content'] = dia_without
        json.dump(chunk_with_labeling, open(path_without, 'w'))
