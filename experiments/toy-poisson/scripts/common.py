import math
import numpy as np


class Preset:
    def __init__(self, args):
        self.rates = np.random.rand(args.n_labels) * args.max_frequency
        self.length = args.length

    def generate(self):
        # 1. Generate L events for each label.
        labels = []
        timestamps = []
        for label, rate in enumerate(self.rates):
            scale = 1 / rate
            timestamps.append(np.cumsum(np.random.exponential(scale, size=self.length)))
            labels.append(np.full([self.length], label))
        # 2. Merge sequences and truncate to the specified length.
        timestamps = np.concatenate(timestamps)
        labels = np.concatenate(labels)
        order = np.argsort(timestamps)
        timestamps = timestamps[order][:self.length].tolist()
        labels = labels[order][:self.length].tolist()
        return timestamps, labels

    def log_like(self, timestamps, labels):
        duration = timestamps[-1]
        timestamps = np.asarray(timestamps)
        labels = np.asarray(labels)
        # Label streams are independent. Log-likelihood is a sum of log-likelihoods for each label.
        l = 0
        for label, rate in enumerate(self.rates):
            mask = labels == label
            if mask.sum() > 0:
                lts = timestamps[mask]
                extended = np.concatenate([np.zeros_like(lts[:1]), lts])
                intertimes = extended[1:] - extended[:-1]
                log_pdfs = math.log(rate) - rate * intertimes
                l += log_pdfs.sum()
                last_time = lts[-1]
            else:
                last_time = 0
            # Add likelihood for the end of a sequence.
            l += -rate * (duration - last_time)
        return l


class Model:
    def __init__(self, args):
        self.presets = [Preset(args) for _ in range(args.n_presets)]

    def generate(self):
        target = int(np.random.choice(len(self.presets)))
        timestamps, labels = self.presets[target].generate()
        return timestamps, labels, target

    def log_likes(self, timestamps, labels):
        return [p.log_like(timestamps, labels) for p in self.presets]
