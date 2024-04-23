CUDA=$1
MODEL=$2
NAME=$3
POOLING=$4
BATCH_SIZE=$5
BENCH=$6

python3 embed_domain_dataset.py \
    --path-in data/benchmarks-$BENCH/sgd/train \
    --path-out data/scatter-analysis/$NAME-embedded/sgd \
    --model $MODEL \
    --pooling $POOLING \
    --batch-size $BATCH_SIZE \
    --cuda $CUDA

python3 filter_dataset_by_length.py \
    --path-in data/scatter-analysis/$NAME-embedded/sgd \
    --path-out data/scatter-analysis/$NAME-filtered/sgd \
    --tokenizer $MODEL \
    --num-shards 1

for AUG in 'insert' 'replace' 'prune' 'shuffle'; do
    python3 embed_domain_dataset.py \
        --path-in data/scatter-analysis/collected/sgd-$AUG-collected \
        --path-out data/scatter-analysis/$NAME-embedded/sgd-$AUG \
        --model $MODEL \
        --pooling $POOLING \
        --batch-size $BATCH_SIZE \
        --cuda $CUDA
done

for AUG in 'insert' 'replace' 'prune' 'shuffle'; do
    python3 filter_dataset_by_length.py \
        --path-in data/scatter-analysis/$NAME-embedded/sgd-$AUG \
        --path-out data/scatter-analysis/$NAME-filtered/sgd-$AUG \
        --tokenizer $MODEL \
        --num-shards 1
done

