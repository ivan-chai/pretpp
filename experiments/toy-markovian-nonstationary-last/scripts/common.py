import numpy as np


class Preset:
    def __init__(self, args):
        probs = np.random.rand(args.n_labels, args.n_labels)
        probs /= np.sum(probs, axis=1, keepdims=True)
        self.probs = probs

    def generate(self, length):
        timestamps = []
        labels = []
        prev_label = np.random.choice(len(self.probs))
        prev_ts = np.random.randint(0, length)
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
        self.length = args.length
        self.presets = [Preset(args) for _ in range(args.n_presets)]

    def generate(self):
        target = 1 + int(np.random.choice(5))
        length = self.length // target
        last_length = self.length - length * (target - 1)
        lengths = [length] * (target - 1) + [last_length]
        timestamps, labels = [], []
        for l in lengths:
            preset = int(np.random.choice(len(self.presets)))
            ts, ls = self.presets[preset].generate(l)
            timestamps.append(ts)
            labels.append(ls)
        timestamps, labels = sum(timestamps, []), sum(labels, [])
        return timestamps, labels, preset
