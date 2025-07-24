import argparse
import math
import os
import pyspark.sql.functions as F
from datasets import load_dataset
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import TimestampType
from random import Random


SEED = 42
VAL_SIZE = 0.1
TEST_SIZE = 0.1


FILENAME = "tianchi_mobile_recommend_train_user.csv"
TEST_DATE = "2014-12-12"


def parse_args():
    parser = argparse.ArgumentParser("Prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def load_transactions(root):
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    # Read data.
    path = os.path.join(root, FILENAME)
    transactions = spark.read.option("header", True).csv(path).select("user_id", "item_id", "behavior_type", "item_category", "time")
    transactions = transactions.select(
        F.col("user_id").cast("int").alias("id"),
        F.unix_timestamp(F.col("time"), format="yyyy-MM-dd HH").alias("timestamps"),
        F.col("item_category").cast("int").alias("labels"),
        F.col("behavior_type").cast("int").alias("types"),
        F.col("item_id").cast("int").alias("items")
    ).cache()
    df = transactions.filter(F.col("timestamps") < F.unix_timestamp(F.lit(TEST_DATE), format="yyyy-MM-dd"))
    df_target = transactions.filter(F.col("timestamps") >= F.unix_timestamp(F.lit(TEST_DATE), format="yyyy-MM-dd"))
    df_target = df_target.select("id").distinct().withColumn("target", F.lit(1))

    df = df.withColumn("timestamps", (F.col("timestamps") - F.unix_timestamp(F.lit("2014-11-18"), format="yyyy-MM-dd")) / (3600 * 24))
    return df.cache(), df_target.cache()


def train_val_test_split(transactions):
    """Select test set from the labeled subset of the dataset."""
    data_ids = {row["id"] for row in transactions.select("id").distinct().collect()}
    data_ids = list(sorted(list(data_ids)))

    Random(SEED).shuffle(data_ids)
    n_clients_test = int(len(data_ids) * TEST_SIZE)
    n_clients_val = int(len(data_ids) * VAL_SIZE)
    test_ids = set(data_ids[:n_clients_test])
    val_ids = set(data_ids[n_clients_test:n_clients_test + n_clients_val])
    train_ids = set(data_ids[n_clients_test + n_clients_val:])

    testset = transactions.filter(transactions["id"].isin(test_ids))
    valset = transactions.filter(transactions["id"].isin(val_ids))
    trainset = transactions.filter(transactions["id"].isin(train_ids))
    return trainset.persist(), valset.persist(), testset.persist()


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("id")).repartition(n_partitions, "id").write.mode("overwrite").parquet(path)


def main(args):
    print("Load")
    transactions, targets = load_transactions(args.root)

    print("Transform")
    transactions = transactions.withColumn("order", F.struct("timestamps", "items"))
    preprocessor = PysparkDataPreprocessor(
        col_id="id",
        col_event_time="order",
        event_time_transformation="none",
        cols_category=["labels", "types", "items"],
        category_transformation="frequency",
        cols_identity=["timestamps"]
    )
    transactions = preprocessor.fit_transform(transactions).drop("order", "event_time")  # id, timestamps, labels, types.
    transactions = transactions.join(targets, on="id", how="left").fillna(0).persist()
    print("N clients:", transactions.count())

    print("Split")
    # Split.
    train, val, test = train_val_test_split(transactions)

    # Dump.
    train_path = os.path.join(args.root, "train.parquet")
    val_path = os.path.join(args.root, "val.parquet")
    test_path = os.path.join(args.root, "test.parquet")
    print(f"Dump train with {train.count()} records to {train_path}")
    dump_parquet(train, train_path, n_partitions=32)
    print(f"Dump val with {val.count()} records to {val_path}")
    dump_parquet(val, val_path, n_partitions=1)
    print(f"Dump test with {test.count()} records to {test_path}")
    dump_parquet(test, test_path, n_partitions=1)
    print("OK")


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    main(parse_args())
