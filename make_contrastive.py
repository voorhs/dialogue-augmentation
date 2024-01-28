# Algorithm:
# 1. collect all json chunks into multiple hf datasets.Dataset
# 2. join Datasets
# 3. filter out those dialogues, that have no positives or negatives that are differ from original dia
#
# output samples must be of the following format:
# cur_res = {
#     'orig': original dialog
#     'pos': list of positive augmentations,
#     'neg': list of negative augmentations,
# }

if __name__ == "__main__":
    from mylib.datamakers.contrastive import main

    path = 'data-2/augmented'
    names_in = ['replace-test', 'replace-test2']
    name_out = 'joined'

    main(path, names_in, name_out)
