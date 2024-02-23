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
    from argparse import ArgumentParser
    from mylib.datamakers.collect import main

    ap = ArgumentParser()
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    ap.add_argument('--path-in', dest='path_in', required=True)
    ap.add_argument('--path-out', dest='path_out', required=True)
    args = ap.parse_args()

    main(args.path_in, args.path_out)
