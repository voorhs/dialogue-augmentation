INIT_FROM=$1
NAME=$2
HF_MODEL=$3
BENCHMARKS=$4
CUDA=$5
LOGGER=$6
MULTILABEL=$7
N_CLASSES=$8

source .venv/bin/activate

python3 downstream_classification.py \
--hf-model $HF_MODEL \
--dataset-path $BENCHMARKS \
--cuda $CUDA \
--logger $LOGGER \
--pooling cls \
--batch-size 32 \
--finetune-layers 3 \
--name $NAME \
--encoder-weights $INIT_FROM \
--n-workers 4 \
--benchmark $MULTILABEL \
--n-classes $N_CLASSES \
--lr-decay False \
--warmup-pct 0 \
--n-epochs 3
