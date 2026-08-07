import argparse
import itertools
import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import torch
import tqdm

from common import Model
from sklearn.cluster import KMeans


def parse_args():
    parser = argparse.ArgumentParser("Download, prepare and dump dataset to a parquet file.")
    parser.add_argument("--root", help="Dataset root", default="data")
    return parser.parse_args()


def load_labels(args, part):
    data = pd.read_parquet(os.path.join(args.root, f"{part}.parquet"))
    return np.stack(data["labels"].values)


def load_targets(args, part):
    data = pd.read_parquet(os.path.join(args.root, f"{part}.parquet"))
    return np.stack(data["target"].values)


def to_transitions(labels, n_labels, chunk_size):
    b, l = labels.shape
    chunks = labels.reshape(-1, chunk_size)  # (B, L).
    transitions = np.zeros([len(chunks), n_labels, n_labels])
    for i, s in enumerate(chunks):
        for s_prev, s_next in zip(s[:-1], s[1:]):
            transitions[i, s_prev, s_next] += 1
    return transitions.reshape(b, l // chunk_size, n_labels, n_labels)


class KMeansPresetPredictor:
    def __init__(self, n_presets, random_state=0):
        self.model = KMeans(n_clusters=n_presets, random_state=0)

    def fit(self, transitions):
        b, n, l, _ = transitions.shape
        self.model.fit(transitions.reshape(b * n, l * l))

    def predict(self, transitions):
        b, n, l, _ = transitions.shape
        return self.model.predict(transitions.reshape(b * n, l * l)).reshape(b, n)

    @property
    def centroids(self):
        return self.model.cluster_centers_

    @property
    def labels(self):
        return self.model.labels_


class SmoothPresetPredictor:
    """Iteratively update reference transition matrices and labels."""
    def __init__(self, n_presets):
        self.n_presets = n_presets
        self.log_presets = None

    def fit(self, transitions):
        b, n, l, _ = transitions.shape
        pretrain = KMeansPresetPredictor(self.n_presets)
        pretrain.fit(transitions)
        presets_logits = torch.from_numpy(pretrain.centroids.reshape(-1, l, l)).float()  # (N, L, L).
        logits = torch.nn.functional.one_hot(torch.from_numpy(pretrain.labels).long(), num_classes=self.n_presets).float()  # (B, N).

        b = b * n
        transitions = torch.from_numpy(transitions.reshape(b, l, l))  # (B, L, L).

        presets_logits.requires_grad = True
        logits.requires_grad = True
        optimizer = torch.optim.Adam([presets_logits, logits], lr=0.1)

        bar = tqdm.tqdm(range(200))
        for _ in bar:
            log_presets = torch.nn.functional.log_softmax(presets_logits, -1)
            log_conditional = (transitions[:, None] * log_presets[None]).sum(-1).sum(-1)  # (B, N).
            log_prod = log_conditional + torch.nn.functional.log_softmax(logits, -1)
            loss = -torch.logsumexp(log_prod, -1).mean(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_description(f"{loss.item():.3f}")
        self.log_presets = log_presets.detach().cpu().numpy()

    def predict(self, transitions):
        assert self.log_presets is not None
        b, n, l, _ = transitions.shape
        scores = self.log_presets[None, None] * transitions[:, :, None]  # (B, _, N, L, L).
        scores = scores.sum(-1).mean(-1)  # (B, _, N).
        labels = scores.argmax(-1)
        assert labels.shape == (b, n)
        return labels


def main(args):
    with open(os.path.join(args.root, "generator.pkl"), "rb") as fp:
        model = pkl.load(fp)
    n_labels = len(model.presets[0].probs)
    n_presets = model.n_presets
    chunk_size = model.chunk_size

    # Reconstruct presets.
    train_labels = load_labels(args, "train")  # (B, L).
    train_transitions = to_transitions(train_labels, n_labels, chunk_size)  # (B, N, L, L).
    b, n, l, _ = train_transitions.shape
    predictor = SmoothPresetPredictor(n_presets)
    predictor.fit(train_transitions)

    # Predict.
    labels_val = load_labels(args, "val")
    targets_val = load_targets(args, "val")

    labels_test = load_labels(args, "test")
    targets_test = load_targets(args, "test")

    labels = np.concatenate([labels_val, labels_test])
    targets = np.concatenate([targets_val, targets_test])

    b = len(labels)
    transitions = to_transitions(labels, n_labels, chunk_size)  # (B, N, L, L).
    b, n, l, _ = transitions.shape
    presets = predictor.predict(transitions)  # (B, N).

    meta_transitions = np.zeros([b, n_presets, n_presets])
    for i, seq in enumerate(presets):
        for p_prev, p_next in zip(seq[:-1], seq[1:]):
            meta_transitions[i, p_prev, p_next] += 1

    predictions = []
    for p, t in zip(presets, meta_transitions):
        r = t[p[-1]]
        if r.sum() == 0:
            predictions.append(random.randrange(n_presets))
        else:
            predictions.append(r.argmax(0))
    predictions = np.array(predictions)

    predictions_val = predictions[:len(labels_val)]
    predictions_test = predictions[len(labels_val):]

    # Find best labels permutation.
    max_accuracy = 0
    best_permutation = None
    for p in itertools.permutations(range(n_presets)):
        p = np.array(p)
        p_predictions = p[predictions_val.flatten()].reshape(*predictions_val.shape)
        accuracy = (p_predictions == targets_val).astype(float).mean()
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_permutation = p
    print("Val accuracy:", max_accuracy)

    p_predictions = best_permutation[predictions_test.flatten()].reshape(*predictions_test.shape)
    accuracy = (p_predictions == targets_test).astype(float).mean()
    print("Test accuracy:", accuracy)


if __name__ == "__main__":
    main(parse_args())
