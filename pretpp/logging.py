import os
import yaml
import warnings
import torch


def log_dict(pl_logger, data, epoch, prefix):
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized() and (torch.distributed.get_world_size() > 1)
    if is_distributed and (torch.distributed.get_rank() != 0):
        return
    logger = pl_logger.experiment
    logger_name = type(logger).__name__
    try:
        if logger_name == "MlflowClient":
            logger.log_dict(pl_logger._run_id, data, f"{prefix}{epoch}.yaml")
            logger.log_dict(pl_logger._run_id, data, f"{prefix}last.yaml")
        elif logger_name == "SummaryWriter":
            # TensorBoard: log YAML content as text.
            text = yaml.dump(data)
            logger.add_text(prefix.rstrip("/"), f"```yaml\n{text}\n```", global_step=epoch)
        elif logger_name == "Run":
            # WandB: write YAML files to the run directory.
            run_dir = getattr(logger, "dir", None)
            if run_dir is not None:
                artifact_dir = os.path.join(run_dir, prefix.rstrip("/"))
                os.makedirs(artifact_dir, exist_ok=True)
                for filename in [f"{epoch}.yaml", "last.yaml"]:
                    with open(os.path.join(artifact_dir, filename), "w") as f:
                        yaml.dump(data, f)
                logger.save(os.path.join(artifact_dir, "*.yaml"), policy="now")
        elif logger_name == "ExperimentWriter":
            # CSV logger: write YAML files alongside CSV logs.
            artifact_dir = os.path.join(logger.log_dir, prefix.rstrip("/"))
            os.makedirs(artifact_dir, exist_ok=True)
            for filename in [f"{epoch}.yaml", "last.yaml"]:
                with open(os.path.join(artifact_dir, filename), "w") as f:
                    yaml.dump(data, f)
        else:
            warnings.warn(f"log_dict: unsupported logger type '{logger_name}', skipping.")
    except Exception as e:
        warnings.warn(f"log_dict: failed to log with {logger_name}: {e}")
