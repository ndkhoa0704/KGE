python main.py --train-path "./data/WN18RR/train.txt.json" \
    --test-path "./data/WN18RR/test.txt.json"  \
    --valid-path "./data/WN18RR/valid.txt.json"  \
    --ents-path "./data/WN18RR/entities.json"  \
    --batch-size 512  \
    --max-word-len 50 \
    --task test