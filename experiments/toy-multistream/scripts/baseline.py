import argparse
import numpy as np
import os
import pandas as pd
import pickle as pkl
from sklearn.metrics import roc_auc_score

from common import Model, Preset


def parse_args():
    parser = argparse.ArgumentParser("Evaluate the Bayesian classifier, i.e. the downstream quality upper bound.")
    parser.add_argument("--root", help="Dataset root", default="data")
    parser.add_argument("--parts", help="Dataset parts to evaluate", nargs="+", default=["val", "test"])
    return parser.parse_args()


def evaluate(model, path):
    data = pd.read_parquet(path)
    scores = np.asarray([model.target_proba(labels, types)
                         for labels, types in zip(data["labels"], data["types"])])
    targets = np.asarray(data["target"])
    return {
        "downstream-auroc": roc_auc_score(targets, scores),
        "downstream-accuracy": ((scores > 0.5) == targets).mean()
    }


def main(args):
    with open(os.path.join(args.root, "generator.pkl"), "rb") as fp:
        model = pkl.load(fp)

    for part in args.parts:
        metrics = evaluate(model, os.path.join(args.root, f"{part}.parquet"))
        for name, value in metrics.items():
            print(f"{part}/{name}: {value:.4f}")


if __name__ == "__main__":
    main(parse_args())
