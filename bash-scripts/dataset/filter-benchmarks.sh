PATH_IN=$1
PATH_OUT=$2
TOKENIZER=$3

for DATASET in "bitod" "multiwoz" "sgd"; do
    for SPLIT in "train" "test" "validation"; do
        python3 filter_dataset_by_length.py \
            --path-in "$PATH_IN/$DATASET/$SPLIT" \
            --path-out "$PATH_OUT/$DATASET/$SPLIT" \
            --tokenizer "$TOKENIZER" \
            --num-shards 1
    done
done
