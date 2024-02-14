if __name__ == "__main__":
    # ==== configure env ====
    import os
    workdir = os.getcwd()
    default_path_in = os.path.join(workdir, 'data', 'source', 'train')

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', dest='method_name', required=True, choices=[
        'back-translate', 'insert', 'replace',
        'listwise-shuffler', 'prune', 'shuffle'
    ])
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    ap.add_argument('--name', dest='name', required=True)
    ap.add_argument('--cuda', dest='cuda', required=True)
    ap.add_argument('--batch-size', dest='batch_size', required=True, type=int)
    ap.add_argument('--path-in', dest='path_in', default=default_path_in)
    ap.add_argument('--skip-existing', dest='skip_existing', action='store_true')
    args = ap.parse_args()

    from mylib.utils.training import init_environment
    init_environment(args)

    # ==== main ====
    from mylib.datamakers.augmentations import main

    main(args)
