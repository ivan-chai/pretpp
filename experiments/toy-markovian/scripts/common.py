import numpy as np


class Preset:
    def __init__(self, args):
        probs = np.random.rand(args.n_labels, args.n_labels)
        probs /= np.sum(probs, axis=1, keepdims=True)
        self.probs = probs
        self.length = args.length

    def generate(self):
        timestamps = []
        labels = []
        prev_label = np.random.choice(len(self.probs))
        prev_ts = np.random.randint(0, self.length)
        for i in range(self.length):
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
        self.presets = [Preset(args) for _ in range(args.n_presets)]

    def generate(self):
        target = int(np.random.choice(len(self.presets)))
        timestamps, labels = self.presets[target].generate()
        return timestamps, labels, target

    def log_likes(self, labels):
        return [p.log_like(labels) for p in self.presets]
