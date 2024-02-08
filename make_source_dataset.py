if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from mylib.datamakers.source import main
    
    ap = ArgumentParser()
    ap.add_argument('--path', dest='path', default=os.path.join(os.getcwd(), 'data', 'source'))
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    args = ap.parse_args()

    main(output_path=args.path, seed=args.seed)
