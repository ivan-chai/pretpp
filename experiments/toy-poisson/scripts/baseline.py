import argparse
import numpy as np
import os
import sys
import pickle as pkl
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

from common import Model, Preset


def parse_args():
    parser = argparse.ArgumentParser("Evaluate Bayesian classifier.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def main(args):
    spark = SparkSession.builder.getOrCreate()
    with open(os.path.join(args.root, "generator.pkl"), "rb") as fp:
        model = pkl.load(fp)

    df = spark.read.parquet(os.path.join(args.root, "test.parquet")).repartition(8).persist()
    predict_udf = F.udf(lambda timestamps, labels: int(np.argmax(model.log_likes(timestamps, labels))), LongType())
    df = df.withColumn("prediction", predict_udf(F.col("timestamps"), F.col("labels")))
    df = df.withColumn("is_correct", F.col("prediction") == F.col("target"))
    results = df.select(F.mean(F.col("is_correct").cast("float")).alias("mean")).collect()
    accuracy = results[0]["mean"]
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    main(parse_args())
