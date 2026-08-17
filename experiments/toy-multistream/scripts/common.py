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
        self.n_streams = args.n_streams
        self.min_length, self.max_length = args.min_length, args.max_length
        self.presets = [Preset(args) for _ in range(args.n_presets)]

    def generate(self):
        preset_indices = [np.random.choice(len(self.presets)) for _ in range(self.n_streams)]
        target = sum(preset_indices) % 2
        length = np.random.randint(self.min_length, self.max_length + 1)

        timestamps = [float(t) for t in range(length)]
        base_labels = [self.presets[i].generate(length)[1] for i in preset_indices]
        labels = []
        types = []
        for i in range(length):
            types.append(i % self.n_streams)
            labels.append(base_labels[types[-1]][i // self.n_streams])
        return timestamps, labels, types, target
