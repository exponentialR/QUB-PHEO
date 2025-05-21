from torch.utils.data import DataLoader
from datasets.dataset import QUBPHEOAerialDataset
from models.models import TwoHeadNet
from utils.metrics_utils import mpjpe_2d, ade_fde_2d, intent_f1, save_md_and_plot

import random
import numpy as np
import torch
import pandas as pd

def build_hand_adjacency():
    # Skeleton connectivity for one hand (MediaPipe 21‐point format):
    # 0: wrist
    # thumb: 1→2→3→4
    # index: 5→6→7→8
    # middle: 9→10→11→12
    # ring: 13→14→15→16
    # pinky: 17→18→19→20
    finger_chains = [
        [0, 1, 2, 3, 4],    # thumb
        [0, 5, 6, 7, 8],    # index
        [0, 9, 10,11,12],   # middle
        [0,13,14,15,16],    # ring
        [0,17,18,19,20],    # pinky
    ]

    # build undirected edges for a single hand
    left_edges = []
    for chain in finger_chains:
        for u, v in zip(chain, chain[1:]):
            left_edges.append((u, v))

    # mirror for right hand (indices 21–41)
    right_edges = [(u+21, v+21) for (u, v) in left_edges]

    # combine
    edges = left_edges + right_edges

    # build adjacency
    A = np.zeros((42, 42), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    # add self-loops
    for i in range(42):
        A[i, i] = 1

    return torch.from_numpy(A)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(cfg):
    data_cfg = cfg["data"].copy()
    for bad in ("obj_box_dim", "sur_box_dim"):
        data_cfg.pop(bad, None)

    dataset_tr = QUBPHEOAerialDataset(split="train", **data_cfg)
    num_classes = len(dataset_tr.subtask2idx)    # returns 36

    labels = [lbl for (_, _, lbl) in dataset_tr.samples]
    counts = np.bincount(labels, minlength=num_classes)
    eps = 1e-6
    class_weights = 1.0 / (counts + eps)
    class_weights = class_weights / class_weights.mean()
    sample_w = class_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_w,
                                                    num_samples=len(dataset_tr),
                                                    replacement=True)
    return sampler, class_weights, dataset_tr


def summarize_model(model):
    """
    Returns a (DataFrame, total_params_M) pair:
     - DataFrame listing each leaf module and its parameter count (in M).
     - total_params_M: sum of all parameters in the model (in M).
    """
    rows = []
    total = 0
    for name, module in model.named_modules():
        # only leaf modules (no children) and with parameters
        if len(list(module.children())) == 0:
            p = sum(p.numel() for p in module.parameters(recurse=False))
            if p > 0:
                rows.append((name or "model", p / 1e6))
                total += p
    df = pd.DataFrame(rows, columns=["Module", "Params (M)"])
    total_M = total / 1e6
    return df, total_M

def print_model_summary(model):
    df, total_M = summarize_model(model)
    print("\nModel parameter summary:")
    print(df.to_markdown(index=False, floatfmt=(".0f", ".3f")))
    print(f"\nTotal parameters: {total_M:.2f} M\n")


def get_loader(split, batch, **kwargs):
    """
    Returns a DataLoader for the given split.
    :param split: Split name (train, val, test)
    :param batch: Batch size
    :param kwargs: Additional arguments for DataLoader
    :return: DataLoader object
    """
    for bad in ("obj_box_dim", "sur_box_dim"):
        kwargs.pop(bad, None)
    qpheo_dataset = QUBPHEOAerialDataset(split=split, **kwargs)
    return DataLoader(qpheo_dataset, batch_size=batch, shuffle=split=='train',
                      num_workers=4, drop_last=True, pin_memory=True)


if __name__ == "__main__":
    # Example usage
    A = build_hand_adjacency()   # shape (42,42), float32
    print("Adjacency matrix shape:", A.shape)
    print("Sample row (joint 0 connections):", A[0].nonzero().flatten().tolist())