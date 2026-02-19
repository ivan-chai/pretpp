import argparse
import math
import os
import pyspark.sql.functions as F
import tarfile
from huggingface_hub import hf_hub_download
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from random import Random


SEED = 42

# Фиксированный порядок месяцев для списков months_*
MONTHS_ORDER = [
    #"2021-01-01","2022-01-01","2022-01-31",
    "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31",
    "2022-06-30", "2022-07-31", "2022-08-31", "2022-09-30",
    "2022-10-31", "2022-11-30", "2022-12-31", "2023-01-31",
]

# Unix-timestamp конца каждого месяца (последняя секунда дня)
import calendar, datetime
MONTHS_TIMESTAMPS = [
    int(calendar.timegm(datetime.datetime.strptime(m, "%Y-%m-%d").timetuple())) + 86399# - 2592000
    for m in MONTHS_ORDER
]


def parse_args():
    parser = argparse.ArgumentParser("Download, prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def make_from_folds(folds, data_root, targets_root, dst, n_partitions):
    spark = SparkSession.builder.getOrCreate()
    dataset = None
    for fold in folds:
        part = spark.read.parquet(os.path.join(data_root, f"fold={fold}"))
        targets = spark.read.parquet(os.path.join(targets_root, f"fold={fold}"))
        targets = targets.drop("is_balanced", "trans_count", "dif_trans_data")
        mon_values = targets.select("mon").distinct().rdd.flatMap(lambda x: x).collect()
        transformed_tables = []
        base_df = targets.select("client_id").distinct()
        for mon_val in mon_values:
            df_mon = targets.filter(F.col("mon") == mon_val)
            df_mon_renamed = df_mon.select(
                "client_id",
                F.col("target_1").alias(f"target_1_{mon_val}"),
                F.col("target_2").alias(f"target_2_{mon_val}"),
                F.col("target_3").alias(f"target_3_{mon_val}"),
                F.col("target_4").alias(f"target_4_{mon_val}"),
            )
            transformed_tables.append(df_mon_renamed)
        result_df = base_df
        for table in transformed_tables:
            result_df = result_df.join(table, on="client_id", how="left")
        targets = result_df
        part = part.join(targets, on="client_id", how="left")
        dataset = dataset.union(part) if dataset is not None else part

    # Compute log_amount.
    udf = F.udf(lambda x: [math.log(abs(v) + 1) for v in x], ArrayType(FloatType(), False))
    dataset = dataset.withColumn("log_amount", udf(F.col("amount")))
    # Scale event_time by 4 days.
    udf = F.udf(lambda x: [v / 345600 for v in x], ArrayType(FloatType(), False))
    dataset = dataset.withColumn("timestamps", udf(F.col("event_time")))

    # --- months_last_id ---
    # Для каждого клиента и каждого из 12 месяцев находим индекс последней операции,
    # совершённой до конца этого месяца (включительно). Если операций не было — -1.
    months_ts = MONTHS_TIMESTAMPS  # захватываем в замыкание

    def compute_months_last_id(event_times):
        result = []
        for month_end_ts in months_ts:
            last_idx = -1
            for i, t in enumerate(event_times):
                if t <= month_end_ts:
                    last_idx = i
            result.append(last_idx)
        return result

    udf_last_id = F.udf(compute_months_last_id, ArrayType(IntegerType(), False))
    dataset = dataset.withColumn("months_last_id", udf_last_id(F.col("event_time")))

    # --- months_target_1..4 ---
    # Берём только те месяцы из MONTHS_ORDER, для которых колонка реально существует в датасете.
    # Это исправляет проблему с nan: если колонки нет — она не читается как null.
    existing_cols = set(dataset.columns)
    actual_months = [m for m in MONTHS_ORDER if f"target_1_{m}" in existing_cols]

    for target_n in range(1, 5):
        col_names = [f"target_{target_n}_{mon}" for mon in actual_months]
        dataset = dataset.withColumn(
            f"months_target_{target_n}",
            F.array(*[F.col(c) for c in col_names])
        )

    # Dump.
    dataset.repartition(n_partitions, "client_id").write.mode("overwrite").parquet(dst)


def load_transactions(root, cache_dir):
    # Download and extract.
    hf_hub_download(repo_id="ai-lab/MBD-mini", filename="ptls.tar.gz", repo_type="dataset", local_dir=cache_dir)
    archive = os.path.join(cache_dir, "ptls.tar.gz")
    with tarfile.open(archive, "r:gz") as fp:
        fp.extractall(path=root)

    hf_hub_download(repo_id="ai-lab/MBD-mini", filename="targets.tar.gz", repo_type="dataset", local_dir=cache_dir)
    archive = os.path.join(cache_dir, "targets.tar.gz")
    with tarfile.open(archive, "r:gz") as fp:
        fp.extractall(path=root)

    # Merge folds.
    make_from_folds([0, 1, 2, 3],
                    os.path.join(root, "ptls", "trx"),
                    os.path.join(root, "targets"),
                    os.path.join(root, "train.parquet"),
                    n_partitions=64)
    make_from_folds([4],
                    os.path.join(root, "ptls", "trx"),
                    os.path.join(root, "targets"),
                    os.path.join(root, "valid.parquet"),
                    n_partitions=16)


def main(args):
    cache_dir = os.path.join(args.root, "cache")
    if not os.path.isdir(args.root):
        os.mkdir(args.root)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    load_transactions(args.root, cache_dir)
    print("OK")


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    #print(MONTHS_TIMESTAMPS)
    main(parse_args())
