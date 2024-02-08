# Algorithm:
# 1. collect all json chunks into multiple datasets.Dataset
# 2. convert them to pyarrow.parquet
# 3. join parquets
# 4. convert back to datasets.Dataset
#
# output samples must be of the following format:
# cur_res = {
#     'orig': original dialog
#     'pos': list of positive augmentations
# }

if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from mylib.datamakers.contrastive import main

    ap = ArgumentParser()
    ap.add_argument('--path', dest='path', default=os.path.join(os.getcwd(), 'data', 'augmented'))
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    ap.add_argument('--directories-in', dest='directories_in', nargs='+', required=True)
    ap.add_argument('--directory-out', dest='directory_out', default='joined')
    args = ap.parse_args()

    main(args.path, args.directories_in, args.directory_out)
