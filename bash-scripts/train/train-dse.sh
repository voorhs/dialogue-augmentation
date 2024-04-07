source .venv/bin/activate

python3 train_baseline_dialogue_encoder.py \
--hf-model aws-ai/dse-bert-base \
--contrastive-path data/train-dse/advanced-light-dse/ \
--multiwoz-path data/benchmarks-dse/multiwoz \
--bitod-path data/benchmarks-dse/bitod \
--sgd-path data/benchmarks-dse/sgd \
--cuda "0,1" \
--logger comet \
--pooling avg \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name dse-advanced-dse-light-mixed-avg \
--emb-dropout 0.1
