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


PURCHASES = "purchases.csv"
CLIENTS = "clients.csv"
PRODUCTS = "products.csv"
TRAIN = "uplift_train.csv"


def parse_args():
    parser = argparse.ArgumentParser("Prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def load_clients_mapping(root):
    spark = SparkSession.builder.getOrCreate()
    path = os.path.join(root, CLIENTS)
    clients = spark.read.option("header", True).csv(path).select("client_id")
    window = Window.orderBy("client_id")
    clients = clients.withColumn("id", F.row_number().over(window))  # client_id, id.
    return clients.cache()  # "client_id", "id".


def log_scale(column):
    c = F.col(column).cast("float")
    return (F.when(c >= 0, F.lit(1)).otherwise(F.lit(-1)) * F.log(F.abs(c) + 1)).alias(column)


def load_transactions(root, products):
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    # Read data.
    path = os.path.join(root, PURCHASES)
    transactions = spark.read.option("header", True).csv(path).select(
        "client_id", "transaction_datetime", "product_id", "store_id",
        "regular_points_received", "express_points_received", "regular_points_spent", "express_points_spent",
        "purchase_sum", "product_quantity", "trn_sum_from_iss", "trn_sum_from_red"
    ).fillna(-1)
    base_timestamp = F.unix_timestamp(F.lit("2018-11-21"), format="yyyy-MM-dd")
    transactions = transactions.select(
        F.col("client_id"),
        ((F.unix_timestamp(F.col("transaction_datetime")) - base_timestamp) / (3600 * 24)).alias("timestamps"),
        log_scale("regular_points_received"),
        log_scale("express_points_received"),
        log_scale("regular_points_spent"),
        log_scale("express_points_spent"),
        log_scale("purchase_sum"),
        log_scale("product_quantity"),
        log_scale("trn_sum_from_iss"),
        log_scale("trn_sum_from_red"),
        F.col("product_id").alias("labels"),
        F.col("store_id")
    )
    transactions = transactions.join(products, on="labels", how="inner")
    return transactions.cache()


def load_products(root):
    spark = SparkSession.builder.getOrCreate()
    path = os.path.join(root, PRODUCTS)
    int_cols = ["segment_id", "is_own_trademark", "is_alcohol"]
    products = spark.read.option("header", True).csv(path).select(
        F.col("product_id").alias("labels"),
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "segment_id",
        "brand_id",
        "vendor_id",
        "is_own_trademark",
        "is_alcohol",
        log_scale("netto")
    ).fillna(-1, subset=int_cols)
    for col in int_cols:
        products = products.withColumn(col, F.col(col).cast("int"))
    return products.cache()


def load_targets(root, clients):
    spark = SparkSession.builder.getOrCreate()
    path = os.path.join(root, TRAIN)
    targets = spark.read.option("header", True).csv(path).select(
        "client_id",
        F.col("treatment_flg").cast("int").alias("treatment"),
        F.col("target").cast("int").alias("target")
    )
    targets = targets.join(clients.select(["client_id", "id"]), on="client_id", how="inner")
    return targets.cache()  # client_id id, treatment, target.


def train_val_test_split(transactions, targets):
    """Select test set from the labeled subset of the dataset."""
    data_ids = {row["client_id"] for row in transactions.select("client_id").distinct().collect()}
    labeled_ids = {row["client_id"] for row in targets.select("client_id").distinct().collect()}
    labeled_ids = data_ids & labeled_ids
    unlabeled_ids = data_ids - labeled_ids

    labeled_ids = list(sorted(list(labeled_ids)))
    Random(SEED).shuffle(labeled_ids)
    n_clients_test = int(len(data_ids) * TEST_SIZE)
    test_ids = set(labeled_ids[-n_clients_test:])
    train_ids = list(sorted(set(labeled_ids[:-n_clients_test]) | unlabeled_ids))
    Random(SEED + 1).shuffle(train_ids)
    n_clients_val = int(len(data_ids) * VAL_SIZE)
    val_ids = set(train_ids[-n_clients_val:])
    train_ids = set(train_ids[:-n_clients_val])

    testset = transactions.filter(transactions["client_id"].isin(test_ids))
    trainset = transactions.filter(transactions["client_id"].isin(train_ids))
    valset = transactions.filter(transactions["client_id"].isin(val_ids))
    return trainset.persist(), valset.persist(), testset.persist()


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("id")).repartition(n_partitions, "id").write.mode("overwrite").parquet(path)


def main(args):
    print("Load")
    clients = load_clients_mapping(args.root)
    products = load_products(args.root)
    targets = load_targets(args.root, clients)
    transactions = load_transactions(args.root, products)

    print("Transform")
    transactions = transactions.withColumn("order", F.struct("timestamps", "labels"))
    preprocessor = PysparkDataPreprocessor(
        col_id="client_id",
        col_event_time="order",
        event_time_transformation="none",
        cols_category=["store_id", "labels", "level_1", "level_2", "level_3", "level_4", "segment_id", "brand_id", "vendor_id", "is_own_trademark", "is_alcohol"],
        category_transformation="frequency",
        cols_identity=["timestamps", "netto", "regular_points_received", "express_points_received", "regular_points_spent", "express_points_spent", "purchase_sum", "product_quantity", "trn_sum_from_iss", "trn_sum_from_red"]
    )
    transactions = preprocessor.fit_transform(transactions).drop("order", "event_time")
    transactions = transactions.join(clients.select(["client_id", "id"]), on="client_id", how="inner")
    transactions = transactions.join(targets.select(["client_id", "treatment", "target"]), on="client_id", how="left")
    transactions = transactions.fillna(-1, subset=["target"]).cache()
    print("N clients:", transactions.count())

    print("Split")
    # Split.
    train, val, test = train_val_test_split(transactions, targets)

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
