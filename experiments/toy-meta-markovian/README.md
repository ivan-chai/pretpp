# Meta-Markovian toy example.

Each chunk is generated according to one of three preset transition matrices and contains 16 events.

Each sequence consists of 16 chunks. The transitions between chunk types are governed by a meta-transition matrix.

The goal is to predict the type of the next chunk after observing the final event in the sequence.

Make dataset:
```sh
spark-submit ./scripts/make-dataset.py
```

Run baseline (Bayesian classifier):
```sh
spark-submit --py-files ./scripts/common.py ./scripts/baseline.py
```
