import argparse
import math
import os
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Prepare and dump dataset to a parquet file.")
    parser.add_argument("src", help="Source embeddings parquet file")
    parser.add_argument("--dst", help="Target embeddings folder", default="embeddings")
    parser.add_argument("--data", help="Data root", default="data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    relevant = np.load(os.path.join(args.data, "input", "relevant_clients.npy"))
    relevant = pd.DataFrame({"id": relevant})

    df = pd.read_parquet(args.src)
    df = relevant.merge(df, on="id", how="left").fillna(0)
    ids = df["id"].to_numpy()
    embeddings = df.drop(columns=["id", "split"]).to_numpy().astype(np.float16)
    print("Shape:", embeddings.shape)
    if not os.path.exists(args.dst):
        os.mkdir(args.dst)
    np.save(os.path.join(args.dst, "client_ids.npy"), ids)
    np.save(os.path.join(args.dst, "embeddings.npy"), embeddings)
    print("OK")
