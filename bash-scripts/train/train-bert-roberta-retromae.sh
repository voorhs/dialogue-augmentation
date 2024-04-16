source .venv/bin/activate

# bert
# python3 train_baseline_dialogue_encoder.py \
# --hf-model google-bert/bert-base-uncased \
# --contrastive-path data/train-bert/trivial/ \
# --multiwoz-path data/benchmarks-bert/multiwoz \
# --bitod-path data/benchmarks-bert/bitod \
# --sgd-path data/benchmarks-bert/sgd \
# --cuda "0,3" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name bert-trivial-heavy-mixed \
# --emb-dropout 0.1

# python3 train_baseline_dialogue_encoder.py \
# --hf-model google-bert/bert-base-uncased \
# --contrastive-path data/train-bert/advanced-light/ \
# --multiwoz-path data/benchmarks-bert/multiwoz \
# --bitod-path data/benchmarks-bert/bitod \
# --sgd-path data/benchmarks-bert/sgd \
# --cuda "0,2" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name bert-advanced-light-mixed \
# --emb-dropout 0.1


python3 train_baseline_dialogue_encoder.py \
--hf-model google-bert/bert-base-uncased \
--contrastive-path data/train-bert/advanced-light-dse/ \
--multiwoz-path data/benchmarks-bert/multiwoz \
--bitod-path data/benchmarks-bert/bitod \
--sgd-path data/benchmarks-bert/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name bert-advanced-light-dse-mixed \
--emb-dropout 0.1

# roberta
# python3 train_baseline_dialogue_encoder.py \
# --hf-model FacebookAI/roberta-base \
# --contrastive-path data/train-roberta/trivial/ \
# --multiwoz-path data/benchmarks-roberta/multiwoz \
# --bitod-path data/benchmarks-roberta/bitod \
# --sgd-path data/benchmarks-roberta/sgd \
# --cuda "0,3" \
# --logger comet \
# --pooling avg \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name roberta-trivial-heavy-mixed \
# --emb-dropout 0.1

# python3 train_baseline_dialogue_encoder.py \
# --hf-model FacebookAI/roberta-base \
# --contrastive-path data/train-roberta/advanced-light/ \
# --multiwoz-path data/benchmarks-roberta/multiwoz \
# --bitod-path data/benchmarks-roberta/bitod \
# --sgd-path data/benchmarks-roberta/sgd \
# --cuda "0,2" \
# --logger comet \
# --pooling avg \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name roberta-advanced-light-mixed \
# --emb-dropout 0.1

# python3 train_baseline_dialogue_encoder.py \
# --hf-model FacebookAI/roberta-base \
# --contrastive-path data/train-roberta/advanced-light-dse/ \
# --multiwoz-path data/benchmarks-roberta/multiwoz \
# --bitod-path data/benchmarks-roberta/bitod \
# --sgd-path data/benchmarks-roberta/sgd \
# --cuda "0,1" \
# --logger comet \
# --pooling avg \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name roberta-advanced-light-dse-mixed \
# --emb-dropout 0.1

# retromae
# python3 train_baseline_dialogue_encoder.py \
# --hf-model Shitao/RetroMAE \
# --contrastive-path data/train-retromae/trivial/ \
# --multiwoz-path data/benchmarks-retromae/multiwoz \
# --bitod-path data/benchmarks-retromae/bitod \
# --sgd-path data/benchmarks-retromae/sgd \
# --cuda "0,3" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name retromae-trivial-heavy-mixed \
# --emb-dropout 0.1

# python3 train_baseline_dialogue_encoder.py \
# --hf-model Shitao/RetroMAE \
# --contrastive-path data/train-retromae/advanced-light/ \
# --multiwoz-path data/benchmarks-retromae/multiwoz \
# --bitod-path data/benchmarks-retromae/bitod \
# --sgd-path data/benchmarks-retromae/sgd \
# --cuda "0,2" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name retromae-advanced-light-mixed \
# --emb-dropout 0.1


python3 train_baseline_dialogue_encoder.py \
--hf-model Shitao/RetroMAE \
--contrastive-path data/train-retromae/advanced-light-dse/ \
--multiwoz-path data/benchmarks-retromae/multiwoz \
--bitod-path data/benchmarks-retromae/bitod \
--sgd-path data/benchmarks-retromae/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name retromae-advanced-light-dse-mixed \
--emb-dropout 0.1


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
--name dse-advanced-light-dse-mixed-avg \
--emb-dropout 0.1
