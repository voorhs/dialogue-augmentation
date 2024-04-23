source .venv/bin/activate

python3 train_baseline_dialogue_encoder.py \
--hf-model BAAI/bge-base-en-v1.5 \
--contrastive-path data/train-bert/advanced-light-dse/ \
--multiwoz-path data/benchmarks-bert/multiwoz \
--bitod-path data/benchmarks-bert/bitod \
--sgd-path data/benchmarks-bert/sgd \
--cuda "2,3" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name bge-advanced-dse-light-mixed \
--emb-dropout 0.1
