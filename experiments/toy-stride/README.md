C presets, N clients.

Each preset generates deterministic sequences with a particular stride.

All sequences have a length equal to 64.

Make dataset:
```sh
spark-submit ./scripts/make-dataset.py
```

Run baseline (Bayesian classifier):
```sh
spark-submit --py-files ./scripts/common.py ./scripts/baseline.py
```
