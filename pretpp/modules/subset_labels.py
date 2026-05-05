import hashlib
import torch


def true_hash(v):
    h = hashlib.sha256(str(v).encode("utf-8")).digest()
    return int.from_bytes(h, "big")


def get_subset_mask(ids, fraction, base=10000, device=None):
    if isinstance(ids, torch.Tensor):
        if device is None:
            device = ids.device
        ids = ids.cpu().tolist()
    hashes = torch.tensor([true_hash(v) % (base + 1) for v in ids], device=device)
    return hashes / base < fraction
