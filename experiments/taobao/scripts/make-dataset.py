import argparse
import os
import pyspark.sql.functions as F
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession, DataFrame
from random import Random


FILENAME = "tianchi_mobile_recommend_train_user.csv"
BASE_DATE = "2014-11-18"
SEED = 42
VAL_SIZE = 0.15

def parse_args():
    parser = argparse.ArgumentParser("Prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def load_transactions(root):
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    path = os.path.join(root, FILENAME)
    transactions = spark.read.option("header", True).csv(path).select(
        "user_id", "item_id", "behavior_type", "item_category", "time"
    )
    transactions = transactions.select(
        F.col("user_id").cast("string").alias("user_id"),
        F.unix_timestamp(F.col("time"), format="yyyy-MM-dd HH").alias("timestamps"),
        F.col("item_category").cast("int").alias("labels"),
        F.col("behavior_type").cast("int").alias("types"),
        F.col("item_id").cast("int").alias("items")
    ).cache()
    return transactions


def extract_data(transactions: DataFrame, user_suffix: str, start_date: str, mid_date: str, end_date: str):
    """Extract a history window and payment targets for the following week.

    Returns (hist, targets) where hist is event-level and targets is per-client.
    """
    base_ts = F.unix_timestamp(F.lit(BASE_DATE), format="yyyy-MM-dd")
    start_ts = F.unix_timestamp(F.lit(start_date), format="yyyy-MM-dd")
    mid_ts = F.unix_timestamp(F.lit(mid_date), format="yyyy-MM-dd")
    end_ts = F.unix_timestamp(F.lit(end_date), format="yyyy-MM-dd")

    hist = (
        transactions
        .filter((F.col("timestamps") >= start_ts) & (F.col("timestamps") < mid_ts))
        .withColumn("id", F.concat(F.col("user_id"), F.lit(user_suffix)))
        .withColumn("timestamps", (F.col("timestamps") - base_ts) / (3600 * 24))
        .drop("user_id")
    )

    targets = (
        transactions
        .filter(
            (F.col("timestamps") >= mid_ts) & (F.col("timestamps") < end_ts) & (F.col("types") == 4)
        )
        .select(F.concat(F.col("user_id"), F.lit(user_suffix)).alias("id"))
        .distinct()
        .withColumn("target", F.lit(1))
    )

    return hist, targets


def train_val_split(train):
    all_ids = sorted(row["id"] for row in train.select("id").collect())
    Random(SEED).shuffle(all_ids)
    n_val = int(len(all_ids) * VAL_SIZE)
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])
    return train.filter(train["id"].isin(train_ids)), train.filter(train["id"].isin(val_ids))


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("id")).repartition(n_partitions, "id").write.mode("overwrite").parquet(path)


def main(args):
    print("Load")
    transactions = load_transactions(args.root)

    print("Extract windows")
    hist1, targets1 = extract_data(transactions, "_1", "2014-11-18", "2014-11-25", "2014-12-02")
    hist2, targets2 = extract_data(transactions, "_2", "2014-11-25", "2014-12-02", "2014-12-09")
    df_train_raw = hist1.union(hist2)
    train_targets = targets1.union(targets2)
    df_test_raw, test_targets = extract_data(transactions, "_3", "2014-12-02", "2014-12-09", "2014-12-16")

    print("Transform")
    df_train_raw = df_train_raw.withColumn("order", F.struct("timestamps", "items"))
    df_test_raw = df_test_raw.withColumn("order", F.struct("timestamps", "items"))

    preprocessor = PysparkDataPreprocessor(
        col_id="id",
        col_event_time="order",
        event_time_transformation="none",
        cols_category=["labels", "types", "items"],
        category_transformation="frequency",
        cols_identity=["timestamps"]
    )
    combined = preprocessor.fit_transform(df_train_raw).drop("order", "event_time")
    combined = combined.join(train_targets, on="id", how="left").fillna(0, subset=["target"]).persist()
    test = preprocessor.transform(df_test_raw).drop("order", "event_time")
    test = test.join(test_targets, on="id", how="left").fillna(0, subset=["target"]).persist()

    print("Split train/val")
    train, val = train_val_split(combined)

    print("N train clients:", train.count())
    print("N val clients:", val.count())
    print("N test clients:", test.count())

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
