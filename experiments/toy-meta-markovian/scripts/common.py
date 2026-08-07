from sys import meta_path

import numpy as np


def gen_transitions(n, temperature=0.1):
    probs = np.random.rand(n, n)
    probs = np.exp(probs / temperature)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


class Preset:
    def __init__(self, args):
        self.probs = gen_transitions(args.n_labels, temperature=args.temperature)

    def generate(self, length):
        timestamps = []
        labels = []
        prev_label = np.random.choice(len(self.probs))
        prev_ts = 0
        for i in range(length):
            prev_ts += 1
            prev_label = np.random.choice(len(self.probs), p=self.probs[prev_label])
            timestamps.append(float(prev_ts))
            labels.append(int(prev_label))
        return timestamps, labels

    def log_like(self, labels):
        l = np.log(1 / len(self.probs))
        for p, n in zip(labels[:-1], labels[1:]):
            l += np.log(self.probs[p, n])
        return l


class Model:
    def __init__(self, args):
        self.n_presets = args.n_presets
        self.chunk_size = args.chunk_size
        self.n_chunks = args.n_chunks
        self.temperature = args.temperature
        self.presets = [Preset(args) for _ in range(args.n_presets)]

    def generate(self):
        meta_probs = gen_transitions(self.n_presets, temperature=self.temperature)
        timestamps, labels = [], []
        presets = []
        last_ts = 0
        for _ in range(self.n_chunks):
            if not presets:
                presets.append(int(np.random.choice(self.n_presets)))
            else:
                presets.append(np.random.choice(self.n_presets, p=meta_probs[presets[-1]]))
            ts, ls = self.presets[presets[-1]].generate(self.chunk_size)
            ts = [t + last_ts for t in ts]
            last_ts = ts[-1]
            timestamps.append(ts)
            labels.append(ls)
        timestamps, labels = sum(timestamps, []), sum(labels, [])
        target = np.random.choice(self.n_presets, p=meta_probs[presets[-1]])
        return timestamps, labels, target, presets, meta_probs
