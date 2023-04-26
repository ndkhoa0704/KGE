# KGE
- Preprocess data
```

TASK = "wn18rr"
python3 -u preprocess.py \
--task "${TASK}" \
--train-path "./data/${TASK}/train.txt" \
--valid-path "./data/${TASK}/valid.txt" \
--test-path "./data/${TASK}/test.txt"
```


- Execution
```
python main.py --train-path "./data/WN18RR/train.txt.json" \
    --test-path "./data/WN18RR/test.txt.json"  \
    --valid-path "./data/WN18RR/valid.txt.json"  \
    --ents-path "./data/WN18RR/entities.json"  \
    --batch-size 512 --no-epoch=1 \
    --max-word-len 7 \
    --task train
```

```
export CUDA_VISIBLE_DEVICES=0
```

```
python prepare_ent.py --train-path "./data/WN18RR/train.txt.json" \
    --test-path "./data/WN18RR/test.txt.json"  \
    --valid-path "./data/WN18RR/valid.txt.json"  \
    --max-word-len 7 \
    --batch-size 64 \
```