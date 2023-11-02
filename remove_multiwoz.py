if __name__ == '__main__':
    import os
    import json
    from tqdm import tqdm

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', dest='path', required=True)
    args = ap.parse_args()

    chunk_names = [fname for fname in os.listdir(args.path) if fname.endswith('.json')]

    for chunk_name in tqdm(chunk_names):
        chunk_path = os.path.join(args.path, chunk_name)
        chunk = json.load(open(chunk_path, 'r'))
        for dia in chunk:
            if dia['source_dataset_name'] == 'MULTIWOZ2_2':
                if dia['content'] is None:
                    # print('already nulled')
                    continue
                print('nulling')
                dia['content'] = None
        json.dump(chunk, open(chunk_path, 'w'))
