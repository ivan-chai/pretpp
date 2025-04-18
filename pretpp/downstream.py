import atexit
import logging
import multiprocessing as mp
import os
import pickle as pkl
import pytorch_lightning as pl
import queue
import sys
import tempfile
import time
import torch
import warnings
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from hotpp.embed import embeddings_to_pandas, GatherMetric
from hotpp.eval_downstream import targets_to_pandas


class FakeDataModule:
    def __init__(self, splits):
        self.splits = splits


def evaluation_worker(config, tasks_queue, results_queue):
    from hotpp.eval_downstream import eval_downstream
    import luigi
    if config.get("log_level", "INFO") not in ["DEBUG", "INFO"]:
        for name in list(logging.root.manager.loggerDict):
            if "luigi" in name:
                logging.getLogger(name).setLevel(logging.ERROR)
        sys.stderr = open(os.devnull,"w")
        sys.stdout = open(os.devnull, "w")
    try:
        while True:
            task = tasks_queue.get()
            if task is None:
                break
            i, step, data_path = task
            with open(data_path, "rb") as fp:
                data = pkl.load(fp)
            embeddings = data["embeddings"]
            targets = data["targets"]
            splits = embeddings["split"].unique()
            scores = eval_downstream(config, None, FakeDataModule(splits), None,
                                     precomputed_embeddings=embeddings,
                                     precomputed_targets=targets)
            metrics = {f"{split}/downstream": mean
                    for split, (mean, std) in scores.items()}
            results_queue.put((i, step, metrics))
    except Exception as e:
        results_queue.put(e)


class PredictTrainer:
    def predict(self, model, datamodule):
        device = next(iter(model.parameters())).device
        results = []
        for x, y in datamodule.predict_dataloader():
            x = x.to(device)
            if y is not None:
                y = y.to(device)
            results.append(model((x, y)))
        return results


class CheckpointSelector:
    def __init__(self, root, maximize):
        self.root = root
        self.maximize = maximize
        self.metrics = {}  # step -> metric.

    def add(self, step, ckpt):
        """Add checkpoint."""
        if self.metrics and (step <= max(self.metrics)):
            raise ValueError(f"Wrong order: {step} <= {max(self.metrics)}")
        torch.save({"state_dict": ckpt}, self.fname(step))
        self.metrics[step] = None

    def remove(self, step):
        """Remove checkpoint."""
        if step not in self.metrics:
            raise KeyError("Unexpected step")
        os.remove(self.fname(step))
        del self.metrics[step]

    def evaluate(self, step, metric_value):
        """Assign metric value."""
        if self.metrics[step] is not None:
            raise ValueError("Repeated evaluation")
        self.metrics[step] = metric_value
        # Keep the best checkpoint and all checkpoints that were not evaluated.
        best_step = self.get_best_step()
        for step, value in list(self.metrics.items()):
            if (value is not None) and (step != best_step):
                self.remove(step)

    def get_best(self):
        """Get best checkpoint."""
        best_step = self.get_best_step()
        return best_step, self.fname(best_step)

    def get_best_step(self):
        if not self.metrics:
            raise FileNotFoundError("No checkpoints found")
        eps = 1e-6
        metrics = {k: v for k, v in self.metrics.items() if v is not None}
        if not metrics:
            # If there are no evaluated checkpoints, return the last one.
            return max(self.metrics)
        best_metric = max(metrics.values()) if self.maximize else min(metrics.values())
        best_steps = {step for step, metric in metrics.items()
                      if abs(metric - best_metric) < eps}
        best_step = min(best_steps)  # The first best step.
        return best_step

    def fname(self, step):
        return os.path.join(self.root, f"checkpoint-{step}.pth")

    def clean(self):
        for step in list(self.metrics):
            self.remove(step)

    def __del__(self):
        self.clean()


class DownstreamEvaluator:
    """Parallel downstream evaluator.

    Args:
        root: The root folder for checkpoints and embeddings.
        downstream_config: Evaluation config for ptls_validation.
        monitor: Metric for selection.
        maximize: Whether to maximize or miminize the monitor metric.
    """

    def __init__(self, root, downstream_config, monitor=None, maximize=False):
        self.config = downstream_config
        self.trainer = PredictTrainer()
        self.root = root
        self.monitor = monitor

        self.tasks_queue = mp.Queue()
        self.results_queue = mp.Queue()
        atexit.register(lambda: self.tasks_queue.put(None))

        self.worker = mp.Process(target=evaluation_worker, args=(self.config, self.tasks_queue, self.results_queue))
        self.worker.start()

        self.finished = False
        self.num_evaluations = 0
        self.last_seen_step = -1
        self.results = []
        self.new_result_index = 0
        if monitor is not None:
            self.checkpoints = CheckpointSelector(root, maximize=maximize)

    def run_async(self, step, embeddings, targets, state_dict):
        """Run evaluation task."""
        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        if step <= self.last_seen_step:
            warnings.warn(f"Unexpected order of steps: {step} <= {self.last_seen_step}. Skipping evaluation.")
            return
        self.last_seen_step = step
        self._assert_alive()
        if self.monitor is not None:
            self.checkpoints.add(step, state_dict)

        i = self.num_evaluations
        data_path = os.path.join(self.root, f"data-{step}.pkl")
        with open(data_path, "wb") as fp:
            pkl.dump({"embeddings": embeddings, "targets": targets}, fp)
        self.tasks_queue.put((i, step, data_path))
        self.num_evaluations += 1
        self._receive()

    def get(self, wait=False):
        """Get available results.
        Returns:
          A tuple of:
            - results: List of dicts with `step` and `metrics` keys. Metrics contain a dictionary of metrics.
            - finished: Whether there are pending tasks or not.
        """
        self._assert_alive()
        self._receive(wait=wait)
        finished = len(self.results) == self.num_evaluations
        results = self.results[self.new_result_index:]
        self.new_result_index = len(self.results)
        return results, finished

    def get_best_checkpoint(self):
        if self.monitor is None:
            raise RuntimeError("The selection metric is unspecified")
        self._receive(wait=True)
        step, checkpoint_path = self.checkpoints.get_best()
        return step, checkpoint_path

    def destroy(self):
        """Stop all subprocesses."""
        if self.finished:
            return
        if self.worker.is_alive():
            self.worker.kill()
        self.tasks_queue.close()
        self.results_queue.close()
        self.finished = True

    def __del__(self):
        self.destroy()

    def _receive(self, wait=False):
        messaged = False
        while True:
            while not self.results_queue.empty():
                result = self.results_queue.get()
                if isinstance(result, Exception):
                    raise result
                i, step, metrics = result
                assert i == len(self.results), "Some results are missing"
                self.results.append({"step": step, "metrics": metrics})
                if self.monitor is not None:
                    if self.monitor in metrics:
                        self.checkpoints.evaluate(step, metrics[self.monitor])
            n_unfinished = self.num_evaluations - len(self.results)
            if wait and (n_unfinished > 0):
                if not messaged:
                    print(f"Waiting {n_unfinished} unfinished evaluation jobs")
                    messaged = True
                time.sleep(1)
            else:
                break

    def _assert_alive(self):
        if self.finished:
            raise ValueError("Evaluator was finished")
        if not self.worker.is_alive():
            try:
                while True:
                    result = self.results_queue.get(block=False)
                    if isinstance(result, Exception):
                        print("Worker exception:", result)
            except queue.Empty:
                pass
            raise RuntimeError(f"Worker has finished with error {self.worker.exitcode}")


class DownstreamCheckpointCallback(pl.callbacks.Checkpoint):
    """Predict all dataloaders and run evaluation asynchronously.

    The callback is of Checkpoint type and will be place at the end of the list.
    """

    SPLIT_TO_LOOP = {
        "train": "fit_loop",
        "val": "validate_loop",
        "test": "test_loop",
        "predict": "predict_loop"
    }

    def __init__(self, root, downstream_config, monitor=None, maximize=False):
        super().__init__()
        self._root = root
        self._config = downstream_config
        self._monitor = monitor
        self._maximize = maximize
        self._evaluator = None
        self._metric = GatherMetric(3)
        self._run_stage = None
        self.best_model_path = None

    def setup(self, trainer, pl_module, stage):
        self._run_stage = stage
        if stage in ("fit", "validate"):
            # setup the predict data even for fit/validate, as we will call it during `on_validation_epoch_end`
            trainer.datamodule.setup("predict")
        if trainer.global_rank == 0:
            self._evaluator = DownstreamEvaluator(self._root, self._config,
                                                  monitor=self._monitor, maximize=self._maximize)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metric_prefix = None if self._run_stage in ["fit"] else "val/"
        self._run_downstream_evaluation(trainer, pl_module,
                                        wait=self._run_stage not in ["fit"],
                                        metric_prefix=metric_prefix)

    def on_test_epoch_end(self, trainer, pl_module):
        metric_prefix = None if self._run_stage in ["fit"] else "test/"
        self._run_downstream_evaluation(trainer, pl_module,
                                        wait=True,
                                        metric_prefix=metric_prefix)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            for result in self._evaluator.get(wait=True)[0]:
                pl_module.logger.log_metrics(result["metrics"], result["step"])
        trainer.strategy.barrier()
        if self._monitor is not None:
            if trainer.global_rank == 0:
                try:
                    best_step, checkpoint_path = self._evaluator.get_best_checkpoint()
                    print(f"Load the best downstream checkpoint from step {best_step}")
                except FileNotFoundError:
                    checkpoint_path = None
            else:
                checkpoint_path = None
            # TODO: broadcast checkpoint in multi-node training.
            self.best_model_path = trainer.strategy.broadcast(checkpoint_path)
            if self.best_model_path is not None:
                pl_module.load_state_dict(torch.load(self.best_model_path)["state_dict"])

    def _run_downstream_evaluation(self, trainer, pl_module, wait=False, metric_prefix=None):
        id_field = trainer.datamodule.id_field
        target_names = trainer.datamodule.train_data.global_target_fields
        splits = self._config.get("data_splits", trainer.datamodule.splits)

        # Predict.
        by_split = {}
        for split in splits:
            self._metric.reset()
            dataloader = getattr(trainer, self.SPLIT_TO_LOOP[split])._data_source.dataloader()
            all_ids = []
            all_embeddings = []
            all_targets = []
            for batch in dataloader:
                data, targets = pl_module._apply_batch_transfer_handler(batch)
                ids = data.payload[id_field]  # (B).

                embeddings = pl_module.embed(data)
                all_ids.append(ids)
                all_embeddings.append(embeddings)
                all_targets.append(torch.stack([targets.payload[name] for name in target_names], -1))  # (B, T).
            ids = torch.cat(all_ids)
            embeddings = torch.cat(all_embeddings)
            targets = torch.cat(all_targets)
            self._metric.update(ids, embeddings, targets)
            by_split[split] = self._metric.compute()

        # Run evaluation.
        if trainer.global_rank == 0:
            embeddings = embeddings_to_pandas(id_field, {k: (v[0].cpu().tolist(), v[1]) for k, v in by_split.items()})
            targets = targets_to_pandas(id_field, target_names, {k: (v[0].cpu().tolist(), v[2]) for k, v in by_split.items()})
            self._evaluator.run_async(pl_module.global_step, embeddings, targets, pl_module.state_dict())
            results = self._evaluator.get(wait=wait)[0]
            if self._run_stage in ["fit"]:
                for result in results:
                    metrics = result["metrics"]
                    if metric_prefix is not None:
                        metrics = {k: v for k, v in metrics.items() if k.startswith(metric_prefix)}
                    if metrics:
                        pl_module.logger.log_metrics(metrics, result["step"])
            elif len(results) > 0:
                metrics = results[-1]["metrics"]
                if metric_prefix is not None:
                    metrics = {k: v for k, v in metrics.items() if k.startswith(metric_prefix)}
                if metrics:
                    pl_module.log_dict(metrics)
        if wait:
            trainer.strategy.barrier()


class DownstreamCallback(pl.callbacks.Callback):
    def __init__(self, root, downstream_config):
        super().__init__()
        self.callback = DownstreamCheckpointCallback(root, downstream_config)

    def setup(self, trainer, pl_module, stage):
        self.callback.setup(trainer, pl_module, stage)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.callback.on_validation_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self.callback.on_test_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self.callback.on_train_end(trainer, pl_module)
