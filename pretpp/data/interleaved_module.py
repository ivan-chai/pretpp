import torch
import math
import pytorch_lightning as pl
from pytorch_lightning.utilities._pytree import _tree_flatten
from pytorch_lightning.utilities.combined_loader import CombinedLoader, _get_iterables_lengths, _ModeIterator
from hotpp.data import PaddedBatch
from hotpp.data.dataset import ShuffledDistributedDataset, DEFAULT_PARALLELIZM
from hotpp.data.module import HotppDataModule, HotppSampler


class InterleavedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, trainset, valset):
        # Skip super init.
        self.trainset = trainset
        self.valset = valset

    #def __len__(self):
    #    return len(self.dataset)

    def __iter__(self):
        while True:
            yield None

    def set_epoch(self, epoch):
        assert hasattr(self.trainset, "epoch")
        self.trainset.epoch = epoch
        assert hasattr(self.valset, "epoch")
        self.valset.epoch = epoch


class _Interleaved(_ModeIterator):
    def __init__(self, iterables, limits=None, val_period=2):
        if val_period < 2:
            raise ValueError("Val period must be >= 2")
        super().__init__(iterables, limits)
        self._val_period = val_period

    def __next__(self):
        assert len(self.iterators) == 2  # train, val.

        if (self.limits is not None) and any([not math.isinf(l) for l in self.limits]):
            raise NotImplementedError("Limits are not implemented.")

        iterator_idx = 1 if self._idx % self._val_period == 0 else 0
        if iterator_idx == 1:
            try:
                out = next(self.iterators[iterator_idx])
            except StopIteration:
                # Recreate validation loader.
                self.iterators[iterator_idx] = iter(self.iterables[iterator_idx])
                out = next(self.iterators[iterator_idx])
        else:
            out = next(self.iterators[iterator_idx])
        for batch in out:  # x / y.
            if isinstance(batch, PaddedBatch):
                batch.payload["_dataloader_idx"] = iterator_idx

        index = self._idx
        self._idx += 1
        return out, index, iterator_idx

    def __len__(self):
        lengths = _get_iterables_lengths(self.iterables)
        train_length = lengths[0]
        if self.limits is not None:
            train_length = min(train_length, self.limits[0])
        n_val = 1 + max(train_length - 1, 0) // (self._val_period - 1)
        return lengths[0] + n_val


class InterleavedLoader(CombinedLoader):
    def __init__(self, train_loader, val_loader, val_period=2):
        self._iterables = [train_loader, val_loader]
        self._flattened, self._spec = _tree_flatten(self.iterables)
        self._iterator = None
        self._limits = None
        self._mode = "max_size_cycle"
        self._val_period = val_period
        self._sampler = None

    @CombinedLoader.sampler.getter
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        self._sampler = value

    def __iter__(self):
        iterator = _Interleaved(self.flattened, self._limits, val_period=self._val_period)
        iter(iterator)
        self._iterator = iterator
        return self

    def _dataset_length(self):
        raise NotImplementedError("Interleaved loader works with iterable datasets.")


class InterleavedDataModule(HotppDataModule):
    """Interleave batches of training and validation set during training.

    The datamodule is used primarily for HPO.
    """
    def __init__(self,
                 train_path=None,
                 train_params=None,
                 val_path=None,
                 val_params=None,
                 test_path=None,
                 test_params=None,
                 val_period=2,
                 **params):
        super().__init__(
            train_path=train_path,
            train_params=train_params,
            val_path=val_path,
            val_params=val_params,
            test_path=test_path,
            test_params=test_params,
            **params
        )
        self._val_period = val_period

    def train_dataloader(self, rank=None, world_size=None):
        rank = self.trainer.local_rank if rank is None else rank
        world_size = self.trainer.world_size if world_size is None else world_size
        loader_params = {"drop_last": True,
                         "pin_memory": torch.cuda.is_available()}
        loader_params.update(self.train_loader_params)
        cache_size = loader_params.pop("cache_size", 4096)
        parallelize = loader_params.pop("parallelize", DEFAULT_PARALLELIZM)
        seed = loader_params.pop("seed", 0)

        train_dataset = ShuffledDistributedDataset(self.train_data, rank=rank, world_size=world_size,
                                                   cache_size=cache_size,
                                                   parallelize=parallelize,
                                                   seed=seed)
        val_dataset = ShuffledDistributedDataset(self.val_data, rank=rank, world_size=world_size,
                                                 cache_size=cache_size,
                                                 parallelize=parallelize,
                                                 seed=seed)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            collate_fn=train_dataset.dataset.collate_fn,
            **loader_params
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            collate_fn=val_dataset.dataset.collate_fn,
            **loader_params
        )
        loader = InterleavedLoader(train_loader, val_loader, val_period=self._val_period)
        loader.__setattr__("sampler", InterleavedSampler(train_dataset, val_dataset))  # Add set_epoch hook.
        return loader

    def val_dataloader(self, rank=None, world_size=None):
        rank = self.trainer.local_rank if rank is None else rank
        world_size = self.trainer.world_size if world_size is None else world_size
        loader_params = {"pin_memory": torch.cuda.is_available()}
        loader_params.update(self.val_loader_params)
        parallelize = loader_params.pop("parallelize", DEFAULT_PARALLELIZM)
        dataset = ShuffledDistributedDataset(self.val_data, rank=rank, world_size=world_size,
                                             parallelize=parallelize)  # Disable shuffle.
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            **loader_params
        )
        return loader

    def test_dataloader(self, rank=None, world_size=None):
        rank = self.trainer.local_rank if rank is None else rank
        world_size = self.trainer.world_size if world_size is None else world_size
        loader_params = {"pin_memory": torch.cuda.is_available()}
        loader_params.update(self.test_loader_params)
        parallelize = loader_params.pop("parallelize", DEFAULT_PARALLELIZM)
        dataset = ShuffledDistributedDataset(self.test_data, rank=rank, world_size=world_size,
                                             parallelize=parallelize)  # Disable shuffle.
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            **loader_params
        )
        return loader
