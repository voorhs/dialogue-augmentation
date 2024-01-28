if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path-in', dest='path_in', required=True)
    ap.add_argument('--path-out', dest='path_out', required=True)
    ap.add_argument('--tokenizer', dest='tokenizer', required=True)
    ap.add_argument('--upper-bound', dest='upper_bound', default=512, type=int)
    args = ap.parse_args()

    import os

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)

    from mylib.datamakers.filter_by_length import main
    main(args.path_in, args.path_out, args.tokenizer, args.upper_bound)
