if __name__ == "__main__":
    from argparse import ArgumentParser
    from mylib.datamakers.domain import main

    ap = ArgumentParser()
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    ap.add_argument('--path', dest='path', required=True)
    ap.add_argument('--one-domain', dest='one_domain', action='store_true')
    args = ap.parse_args()

    main(args.path, args.one_domain, seed=0)
