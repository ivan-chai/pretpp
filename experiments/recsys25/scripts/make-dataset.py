import argparse
import math
import os
import pyspark.sql.functions as F
import numpy as np
from datasets import load_dataset
from ptls.preprocessing import PysparkDataPreprocessor
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import ArrayType, IntegerType
from random import Random


SEED = 42
VAL_PERC = 5

BUY_TYPE = 0
ADD_TYPE = 1
REM_TYPE = 2

BASE_DATE = "2022-06-23 00:00:00"
TRAIN_DATE = "2022-10-13 00:00:00"
VAL_DATE = "2022-10-27 00:00:00"
DATE_FMT = "yyyy-MM-dd HH:mm:ss"


def parse_args():
    parser = argparse.ArgumentParser("Prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root (must contain MIMIC4 folder with `core`, `hosp`, and `icu`)", default="data")
    return parser.parse_args()


def parse_timestamp(df):
    df = df.withColumn("timestamp", (F.col("timestamp") / 10**9).cast("long"))
    return df


def time_to_day(df, field="timestamps"):
    df = df.withColumn(field,
                       (F.col(field)
                        - F.unix_timestamp(F.lit(BASE_DATE), format=DATE_FMT)) / (3600 * 24))
    return df


def load_target_clients(root):
    spark = SparkSession.builder.getOrCreate()
    ids = np.load(os.path.join(root, "input", "relevant_clients.npy"))
    df = spark.createDataFrame(ids[:, None].tolist(), schema=["id"])
    return df


def load_target_items(root):
    spark = SparkSession.builder.getOrCreate()
    ids = np.load(os.path.join(root, "target", "propensity_sku.npy"))
    df = spark.createDataFrame(ids[:, None].tolist(), schema=["items"])
    return df


def load_target_categories(root):
    spark = SparkSession.builder.getOrCreate()
    ids = np.load(os.path.join(root, "target", "propensity_category.npy"))
    df = spark.createDataFrame(ids[:, None].tolist(), schema=["labels"])
    return df


def load_active_clients(root):
    spark = SparkSession.builder.getOrCreate()
    ids = np.load(os.path.join(root, "target", "active_clients.npy"))
    df = spark.createDataFrame(ids[:, None].tolist(), schema=["id"])
    return df


def load_targets(root, all_ids, target_ids, products):
    spark = SparkSession.builder.getOrCreate()

    target_items = load_target_items(root)
    target_labels = load_target_categories(root)
    active_clients = load_active_clients(root)

    buy = parse_timestamp(spark.read.parquet(os.path.join(root, "product_buy.parquet"))).filter(
        (F.col("timestamp") >= F.unix_timestamp(F.lit(TRAIN_DATE), format=DATE_FMT)) &
        (F.col("timestamp") < F.unix_timestamp(F.lit(VAL_DATE), format=DATE_FMT))
        ).selectExpr(
        "client_id as id",
        "sku as items").distinct()
    buy = buy.join(products.select("items", "labels"), on="items", how="inner")  # id, items, labels.

    targets = all_ids.join(target_ids, on="id", how="left_anti").cache()  # id.

    # Items.
    items = buy.select("id", "items").join(target_items, on="items", how="inner").select("id").distinct()
    targets = targets.join(items.withColumn("target_itemprop", F.lit(1)), on="id", how="left").fillna(0)

    # Categories.
    cats = buy.select("id", "labels").join(target_labels, on="labels", how="inner").select("id").distinct()
    targets = targets.join(cats.withColumn("target_catprop", F.lit(1)), on="id", how="left").fillna(0)

    # Churn.
    churn = buy.select("id").distinct()
    targets = targets.join(churn.withColumn("target_churn_raw", F.lit(0)), on="id", how="left").fillna(1)
    targets = targets.join(active_clients.withColumn("active", F.lit(True)), on="id", how="left").fillna(False)
    targets = targets.withColumn("target_churn", F.when(F.col("active"), F.col("target_churn_raw")).otherwise(None)).drop("target_churn_raw").drop("active")

    return targets.cache()  # id, target_itemprop, target_catprop, target_churn.


def load_products(root):
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.parquet(os.path.join(root, "product_properties.parquet")).selectExpr(
        "sku as items",
        "category as labels",
        "price",
        "name"
    )

    udf = F.udf(lambda s: list(map(int, s.strip("[]").split())), ArrayType(IntegerType()))
    df = df.withColumn("name", udf(F.col("name")))
    columns = ["items", "labels", "price"]
    columns = columns + [F.col("name").getItem(i).alias(f"name_{i:02d}") for i in range(16)]
    df = df.select(columns)
    return df


def load_items(path, source_type, products):
    spark = SparkSession.builder.getOrCreate()

    df = parse_timestamp(spark.read.parquet(path)).filter(
        F.col("timestamp") < F.unix_timestamp(F.lit(TRAIN_DATE), format=DATE_FMT)
        ).selectExpr(
        "client_id as id",
        "timestamp as timestamps",
        "sku as items")
    df = time_to_day(df)
    df = df.join(products, on="items", how="inner")
    df = df.withColumn("types", F.lit(source_type))
    return df.cache()  # id, timestamps, labels, items, price, name_N, types.


def train_val_test_split(transactions, all_ids, target_ids):
    """Split dataset into train / val / test.

    The train part contains all target ids and some non-target
    ids. Last 28 days are excluded for evaluation. Non-targets have
    an assigned label, extracted using 14 days after the last event.

    Validation is a subset of users, not present in
    the target set. The validation set has an assigned label.

    The test is just a subset of train with target clients for prediction.
    """
    non_target_ids = all_ids.join(target_ids, on="id", how="left_anti")
    m = (math.sqrt(5) - 1) / 2
    def mhash(col):
        return 1000000 * ((m * col) % 1)
    non_target_ids = non_target_ids.withColumn("is_val", mhash(F.col("id")) % 100 < VAL_PERC)
    val_ids = non_target_ids.filter(F.col("is_val")).drop("is_val")
    train_ids = non_target_ids.filter(~F.col("is_val")).drop("is_val").union(target_ids)
    print("NT", non_target_ids.count())
    print("VAL", val_ids.count())
    print("TR", train_ids.count())

    trainset = transactions.join(train_ids, on="id", how="inner")
    valset = transactions.join(val_ids, on="id", how="inner")
    testset = trainset.join(target_ids, on="id", how="inner")
    return trainset.persist(), valset.persist(), testset.persist()


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("id")).repartition(n_partitions, "id").write.mode("overwrite").parquet(path)


def main(args):
    products = load_products(args.root)
    items_buy = load_items(os.path.join(args.root, "product_buy.parquet"), BUY_TYPE, products)
    items_add = load_items(os.path.join(args.root, "add_to_cart.parquet"), ADD_TYPE, products)
    items_remove = load_items(os.path.join(args.root, "remove_from_cart.parquet"), REM_TYPE, products)
    target_ids = load_target_clients(args.root)  # id.

    # Join.
    transactions = items_buy.union(items_add).union(items_remove)  # id, timestamps, labels, items, price, name_N, types.
    all_ids = transactions.select("id").distinct().cache()
    print("Total IDS", all_ids.count())

    targets = load_targets(args.root, all_ids, target_ids, products)  # id, target_catprop, target_itemprop, target_churn

    # Group.
    transactions = transactions.withColumn("order", F.struct("timestamps", "items"))

    print("Transform")
    preprocessor = PysparkDataPreprocessor(
        col_id="id",
        col_event_time="order",
        event_time_transformation="none",
        cols_category=["labels", "items"],
        category_transformation="frequency",
        cols_identity=["timestamps", "types", "price"] + [f"name_{i:02d}" for i in range(16)]
    )
    transactions = preprocessor.fit_transform(transactions).drop("order", "event_time")  # id, timestamps, labels, types.
    transactions = transactions.join(targets, on="id", how="left").persist()

    print("Split")
    # Split.
    train, val, test = train_val_test_split(transactions, all_ids, target_ids)

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
    spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.sparkContext.setLogLevel("WARN")
    main(parse_args())
