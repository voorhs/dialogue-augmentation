if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path-in', dest='path_in', required=True)
    ap.add_argument('--path-out', dest='path_out', required=True)
    ap.add_argument('--num-shards', dest='num_shards', default=64, type=int)
    ap.add_argument('--remove', dest='remove', nargs='+', required=True)
    args = ap.parse_args()

    import os

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)

    from mylib.datamakers.filter_by_aug import main
    main(args.path_in, args.path_out, args.remove, args.num_shards)
