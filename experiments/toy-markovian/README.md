C presets, N clients.

Each preset generates sequences according to some transition probabilities matrix. The goal is to recognize preset from each client's transactions.

Make dataset:
```sh
spark-submit ./scripts/make-dataset.py
```

Run baseline (Bayesian classifier):
```sh
spark-submit --py-files ./scripts/common.py ./scripts/baseline.py
```
