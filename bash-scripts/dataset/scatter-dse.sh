CUDA=$1
NAME="DSE"
POOLING="avg"
BATCH_SIZE=96
BENCH="bert"

bash bash-scripts/dataset/scatter-analysis.sh \
    $CUDA \
    aws-ai/dse-bert-base \
    $NAME \
    $POOLING \
    $BATCH_SIZE \
    $BENCH


for EPOCH in '0' '1' '2' '3' '4'; do
    bash bash-scripts/dataset/scatter-analysis.sh \
        $CUDA \
        "pretrained/dse/advanced-light-dse-mixed/epoch=$EPOCH" \
        "$NAME-trained-$EPOCH" \
        $POOLING \
        $BATCH_SIZE \
        $BENCH
done