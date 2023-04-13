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
python main.py --train-path "./data/WN18RR/test.txt.json" --test-path "./data/WN18RR/valid.txt.json"  --batch-size 512 --no-epoch=10
```