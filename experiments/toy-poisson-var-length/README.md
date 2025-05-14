C presets, N clients.

Each preset is characterized by label frequencies. The goal is to recognize preset from each client's transactions.

Sequences have varying lengths.

Make dataset:
```sh
spark-submit ./scripts/make-dataset.py
```

Run baseline (Bayesian classifier):
```sh
spark-submit --py-files ./scripts/common.py ./scripts/baseline.py
```
