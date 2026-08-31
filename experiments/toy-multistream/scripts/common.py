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
        if len(labels) == 0:
            return 0.0
        # The first label follows a single transition from a uniform initial state.
        l = np.log(self.probs[:, labels[0]].mean())
        for p, n in zip(labels[:-1], labels[1:]):
            l += np.log(self.probs[p, n])
        return l


class Model:
    def __init__(self, args):
        self.n_streams = args.n_streams
        self.min_length, self.max_length = args.min_length, args.max_length
        self.presets = [Preset(args) for _ in range(args.n_presets)]
        # Each (stream, preset) pair contributes a scalar weight to the target score. The target is
        # monotone in each weight, so partial knowledge of a preset improves prediction, unlike a
        # parity target, which is an arbitrary lookup and stays at chance until presets are certain.
        # Weights depend on the stream, so the model must attribute events to the correct stream.
        weights = np.random.randn(args.n_streams, args.n_presets)
        self.weights = weights - weights.mean(axis=1, keepdims=True)  # Balance the target classes.

    def score(self, preset_indices):
        return sum(self.weights[stream, preset] for stream, preset in enumerate(preset_indices))

    def generate(self):
        preset_indices = [np.random.choice(len(self.presets)) for _ in range(self.n_streams)]
        target = int(self.score(preset_indices) > 0)
        length = np.random.randint(self.min_length, self.max_length + 1)

        timestamps = [float(t) for t in range(length)]
        base_labels = [self.presets[i].generate(length)[1] for i in preset_indices]
        labels = []
        types = []
        for i in range(length):
            types.append(i % self.n_streams)
            labels.append(base_labels[types[-1]][i // self.n_streams])
        return timestamps, labels, types, target

    def split_streams(self, labels, types):
        """Split interleaved labels into per-stream label sequences."""
        streams = [[] for _ in range(self.n_streams)]
        for label, type in zip(labels, types):
            streams[type].append(int(label))
        return streams

    def preset_posterior(self, labels):
        """Compute the posterior distribution over presets for a single stream."""
        log_likes = np.asarray([preset.log_like(labels) for preset in self.presets])
        # The preset prior is uniform, so the posterior is the normalized likelihood.
        posterior = np.exp(log_likes - log_likes.max())
        return posterior / posterior.sum()

    def target_proba(self, labels, types):
        """Compute the Bayesian posterior probability of the target being 1.

        Streams are independent given the observations, so the distribution of the target score is
        the convolution of the per-stream weight distributions. The support grows as
        n_presets ** n_streams, which is small for the toy setup.
        """
        scores = np.zeros(1)
        probas = np.ones(1)
        for stream, stream_labels in enumerate(self.split_streams(labels, types)):
            posterior = self.preset_posterior(stream_labels)
            scores = (scores[:, None] + self.weights[stream][None]).ravel()
            probas = (probas[:, None] * posterior[None]).ravel()
        return float(probas[scores > 0].sum())
