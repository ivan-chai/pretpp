import numpy as np


class Preset:
    def __init__(self, args):
        self.stride = np.random.rand() * 10
        self.length = args.length

    def generate(self):
        timestamps = [np.random.rand() * 10]
        labels = [0]
        for i in range(self.length):
            timestamps.append(timestamps[-1] + self.stride)
            labels.append(0)
        return timestamps, labels


class Model:
    def __init__(self, args):
        self.presets = [Preset(args) for _ in range(args.n_presets)]

    def generate(self):
        target = int(np.random.choice(len(self.presets)))
        timestamps, labels = self.presets[target].generate()
        return timestamps, labels, target
