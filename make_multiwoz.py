if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    from mylib.datamakers.multiwoz import main

    ap = ArgumentParser()
    ap.add_argument('--path', dest='path', default=os.path.join(os.getcwd(), 'data', 'multiwoz'))
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    args = ap.parse_args()

    main(args.path, seed=0)
