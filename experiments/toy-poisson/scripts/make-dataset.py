import argparse
import numpy as np
import os
import pickle as pkl
import pyspark.sql.functions as F
from random import Random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, LongType, FloatType

from common import Model


SEED = 42
VAL_SIZE = 0.1
TEST_SIZE = 0.1


def parse_args():
    parser = argparse.ArgumentParser("Download, prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    parser.add_argument("--n-presets", help="The number of presets", type=int, default=10)
    parser.add_argument("--n-labels", help="The number of labels", type=int, default=3)
    parser.add_argument("--max-frequency", help="The maximum events frequency", type=float, default=1)
    parser.add_argument("--size", help="Dataset size", type=int, default=10000)
    parser.add_argument("--length", help="Sequence length", type=int, default=64)
    return parser.parse_args()


def dump_parquet(df, path, n_partitions):
    df.sort(F.col("id")).repartition(n_partitions, "id").write.mode("overwrite").parquet(path)


def train_val_test_split(df):
    """Select test set from the labeled subset of the dataset."""
    data_ids = {row["id"] for row in df.select("id").distinct().collect()}
    data_ids = list(sorted(list(data_ids)))
    Random(SEED).shuffle(data_ids)
    n_test = int(len(data_ids) * TEST_SIZE)
    n_val = int(len(data_ids) * VAL_SIZE)

    test_ids = data_ids[:n_test]
    val_ids = data_ids[n_test:n_test + n_val]
    train_ids = data_ids[n_test + n_val:]

    testset = df.filter(df["id"].isin(test_ids))
    valset = df.filter(df["id"].isin(val_ids))
    trainset = df.filter(df["id"].isin(train_ids))
    return trainset.persist(), valset.persist(), testset.persist()


def main(args):
    spark = SparkSession.builder.getOrCreate()
    if not os.path.isdir(args.root):
        os.mkdir(args.root)

    print("Make")
    np.random.seed(0)
    model = Model(args)

    with open(os.path.join(args.root, "generator.pkl"), "wb") as fp:
        pkl.dump(model, fp)

    df = spark.createDataFrame(
        [(i, timestamps, labels, target)
         for i, (timestamps, labels, target) in enumerate([model.generate() for _ in range(args.size)])],
        StructType([
            StructField("id", LongType(), False),
            StructField("timestamps", ArrayType(FloatType(), False), False),
            StructField("labels", ArrayType(LongType(), False), False),
            StructField("target", LongType(), False)
        ])
    )

    print("Split")
    train, val, test = train_val_test_split(df)

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
