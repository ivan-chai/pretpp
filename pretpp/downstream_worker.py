import atexit
import json
import logging
import multiprocessing as mp
import os
import pickle as pkl
import sys
from omegaconf import OmegaConf


class FakeDataModule:
    def __init__(self, splits):
        self.splits = splits


def evaluation_worker(config, tasks_queue, results_queue):
    """Evaluation worker, that operates with queues."""
    if config.get("log_level", "INFO") not in ["DEBUG", "INFO"]:
        for name in list(logging.root.manager.loggerDict):
            if "luigi" in name:
                logging.getLogger(name).setLevel(logging.ERROR)
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")
    else:
        sys.stdout = sys.stderr
    from hotpp.eval_downstream import eval_downstream
    try:
        while True:
            task = tasks_queue.get()
            if task is None:
                break
            i, step, data_path = task
            with open(data_path, "rb") as fp:
                data = pkl.load(fp)
            os.remove(data_path)
            embeddings = data["embeddings"]
            targets = data["targets"]
            splits = embeddings["split"].unique()
            scores = eval_downstream(config, None, FakeDataModule(splits), None,
                                     precomputed_embeddings=embeddings,
                                     precomputed_targets=targets)
            metrics = {f"{split}/{metric}": value
                       for split, metrics in scores.items()
                       for metric, value in metrics.items()}
            results_queue.put((i, step, metrics))
    except Exception as e:
        results_queue.put({"error": str(e)})


def evaluation_subprocess():
    """Evaluation subprocess that communicates with STDIN/STDOUT."""
    config = json.loads(sys.stdin.readline())
    assert isinstance(config, dict)
    config = OmegaConf.create(config)

    tasks_queue = mp.Queue()
    results_queue = mp.Queue()
    atexit.register(lambda: tasks_queue.put(None))

    worker = mp.Process(target=evaluation_worker, args=(config, tasks_queue, results_queue))
    worker.start()
    try:
        for line in sys.stdin:
            task = json.loads(line)
            tasks_queue.put(task)
            if task is None:
                break
            result = results_queue.get(block=True)
            print(json.dumps(result, indent=None, separators=(",", ":")), flush=True)
            if "error" in result:
                break
    except Exception as e:
        result = {"error": str(e)}
        print(json.dumps(result, indent=None, separators=(",", ":")), flush=True)
    finally:
        tasks_queue.close()
        results_queue.close()
        if worker.is_alive():
            worker.kill()


if __name__ == "__main__":
    evaluation_subprocess()
