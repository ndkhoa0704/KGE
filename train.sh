python main.py --train-path "./data/WN18RR/train.txt.json" \
    --test-path "./data/WN18RR/test.txt.json"  \
    --valid-path "./data/WN18RR/valid.txt.json"  \
    --ents-path "./data/WN18RR/entities.json"  \
    --batch-size 32 --no-epoch=1 \
    --max-word-len 50 \
    --task train --learning-rate 0.05 --margin 0.9