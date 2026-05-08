import os


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def get_workdir(logger, make=False):
    root = logger.root_dir or logger.save_dir or "lightning_logs"
    if make:
        safe_mkdir(root)
    version = logger.version
    root = os.path.join(root, version if isinstance(version, str) else f"version_{version}")
    if make:
        safe_mkdir(root)
    return root
