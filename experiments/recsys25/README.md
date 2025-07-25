Homepage:

https://recsys.synerise.com/summary

# Dataprepartion

1. Download the data from the official website and extract data to the "data" folder.

https://data.recsys.synerise.com/dataset/ubc_data/ubc_data.tar.gz

2. Split the data using the official script:

https://github.com/Synerise/recsys2025/blob/main/data_utils/split_data.py

```bash
python -m data_utils.split_data --challenge-data-dir <your_challenge_data_dir>
```

3. Run the data preparation script:

```bash
spark-submit <spark-options> scripts/make-dataset.py
```

## Useful Spark options
Set memory limit:
```bash
spark-submit --driver-memory 6g
```

Set the number of threads:
```bash
spark-submit --master 'local[8]'
```
# Train
```bash
python3 -m hotpp.train --config-dir configs --config-name next_item
```

# Embed
Apply embedder:

```bash
python -m hotpp.embed --config-dir configs --config-name next_item "~logger" "~data_module.train_path" "~data_module.val_path" data_module.test_path=data/test.parquet +embeddings_path=results/embeddings_next_item.parquet
```

Parse outputs:

```bash
python scripts/parse-embeddings.py results/embeddings_next_item.parquet
```

# Evaluate
Use the official RecSys25 Challenge script:

```bash
python -m training_pipeline.train --data-dir <path_to_data> --embeddings-dir <path_to_embeddings> --tasks churn propensity_category propensity_sku --log-name <experiment> --accelerator gpu --devices 0 --disable-relevant-clients-check
```
