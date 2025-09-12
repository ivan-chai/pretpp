#!/usr/bin/env python3
import os
import pyarrow as pa
import tempfile
import torch
from unittest import TestCase, main

from pretpp.data import InterleavedDataModule


def gather_distributed_dataset(data, split, world_size=1, epoch=1):
    get_loader_fn = getattr(data, f"{split}_dataloader")
    loaders = [get_loader_fn(rank=i, world_size=world_size)
               for i in range(world_size)]
    for loader in loaders:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
    by_worker = list(map(list, loaders))
    return by_worker


class TestInterleavedLoader(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

        val_size = 9
        ids = pa.array(list(range(val_size)))
        timestamps = pa.array([list(range(i)) for i in range(val_size)])
        table = pa.Table.from_arrays([ids, timestamps], names=["id", "timestamps"])
        self.val_path = os.path.join(self.root, "val.parquet")
        pa.parquet.write_table(table, self.val_path)

        train_size = 18
        ids = pa.array(list(range(100, 100 + train_size)))
        timestamps = pa.array([list(range(i)) for i in range(train_size)])
        table = pa.Table.from_arrays([ids, timestamps], names=["id", "timestamps"])
        self.train_path = os.path.join(self.root, "train.parquet")
        pa.parquet.write_table(table, self.train_path)

    def tearDown(self):
        """ Called after every test. """
        self.tmp.cleanup()

    def test_interleaved_loader(self):
        for num_workers in [1, 2]:
            data = InterleavedDataModule(train_path=self.train_path,
                                        train_params={
                                            "batch_size": 4,
                                            "num_workers": num_workers,
                                            "cache_size": None  # Disable shuffle.
                                        },
                                        val_path=self.val_path,
                                        val_params={  # Ignored, replaced by train_params.
                                            "batch_size": 3,
                                            "num_workers": num_workers,
                                            "cache_size": None
                                        })
            items = gather_distributed_dataset(data, "train")
            items = sum(items, [])
            ids = torch.cat([v.payload["id"] for (v, _), _, _ in items]).tolist()
            batch_ids = [batch_idx for _, batch_idx, _ in items]
            loader_ids = [dataloader_idx for _, _, dataloader_idx in items]
            self.assertEqual(len(ids), 9 * 4)
            self.assertEqual(set(ids), (set(range(9)) | set(range(100, 100 + 18))) - {8, 116, 117})
            self.assertEqual(batch_ids, list(range(9)))
            self.assertEqual(loader_ids, [1, 0, 1, 0, 1, 0, 1, 0, 1])


if __name__ == "__main__":
    main()
