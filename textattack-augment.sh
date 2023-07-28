textattack augment\
    --input-csv utterances.csv\
    --output-csv augmented.csv\
    --input-column text\
    --recipe embedding\
    --pct-words-to-swap .1\
    --transformations-per-example 1\
    --exclude-original