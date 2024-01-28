# Algorithm:
# 1. collect all json chunks into multiple datasets.Dataset
# 2. covert them to pyarrow.parquet
# 3. join parquets
# 4. convert back to datasets.Dataset
#
# output samples must be of the following format:
# cur_res = {
#     'orig': original dialog
#     'pos': list of positive augmentations
# }

if __name__ == "__main__":
    from mylib.datamakers.contrastive import main

    path = 'data-2/augmented'
    names_in = ['replace-test', 'replace-test2']
    name_out = 'joined'

    main(path, names_in, name_out)
