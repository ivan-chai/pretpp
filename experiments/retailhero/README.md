# Description
The dataset includes purchases and client/products meta-data. The goal is to predict reaction to advertising.

NOTE: the treatment flag is attached to an embedding before training a boosting model.

# Data preparation

Get data:
```bash
sh ./scripts/get-data.sh
```

Preprocess:
```bash
spark-submit <spark-options> scripts/make-dataset.py
```

### Useful Spark options
Set memory limit:
```
spark-submit --driver-memory 6g
```

Set the number of threads:
```
spark-submit --master 'local[8]'
```
