import argparse, yaml, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from datasets.dataset import QUBPHEOAerialDataset
from models.models import TwoHeadNet
from utils.metrics_utils import mpjpe_2d, ade_fde_2d, intent_f1, save_md_and_plot
from models.helper_utils import build_hand_adjacency
import time

import random
import numpy as np
import torch
import pandas as pd

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

def run_epoch(nnet, loader, opt, cfg, ce, mse, device, train=True):
    if train: nnet.train()
    else: nnet.eval()

    tots = {'mpjpe': 0, 'ade': 0, 'fde': 0, 'f1': 0, 'n': 0}

    for batch in loader:
        if cfg['arch'].lower() == 'stgcn':
            x = batch['obs_h'].to(device)  # (B,60,84)
        else:
            parts = [batch['obs_h'].to(device)]  # (B,60,84)
            if 'obs_gaze' in batch:       parts.append(batch['obs_gaze'].to(device))
            if 'obs_obj_box' in batch:    parts.append(batch['obs_obj_box'].to(device))
            if 'obs_sur_box' in batch:    parts.append(batch['obs_sur_box'].to(device))

            x = torch.cat(parts, dim=-1).to(device)  # (B,60,84+2+4+4) provided all given
        y_true = batch['pred_h'].to(device)  # (B,60,84)

        lbl = batch['subtask'].to(device)  # (B,)

        if train: opt.zero_grad()
        yhat, logit = nnet(x)
        loss = cfg['loss']['pose'] * mse(yhat, y_true) + cfg['loss']['intent'] * ce(logit, lbl)

        if train: loss.backward()
        nn.utils.clip_grad_norm_(nnet.parameters(), cfg.get('grad_clip', 1.0))
        opt.step()

        with torch.no_grad():
            b = x.size(0)
            tots["mpjpe"] += mpjpe_2d(yhat, y_true) * b
            ade, fde = ade_fde_2d(yhat, y_true)
            tots["ade"] += ade * b
            tots["fde"] += fde * b
            tots["f1"] += intent_f1(logit, lbl) * b
            tots["n"] += b

    for k in ("mpjpe", "ade", "fde", "f1"):
        tots[k] /= tots["n"]
    return tots

def main(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ADJACENCY MATRIX
    A = build_hand_adjacency()

    feature_dim = 84
    if cfg["data"]["include_gaze"]:      feature_dim += 2
    if cfg["data"]["include_obj_bbox"]:  feature_dim += cfg["data"]["obj_box_dim"]  # e.g. 30*4
    if cfg["data"]["include_surrogate_bbox"]:
        feature_dim += cfg["data"]["sur_box_dim"]  # e.g. 4

    net = TwoHeadNet(
        arch=cfg["arch"],
        input_dim=feature_dim,  # <— new arg
        dropout=cfg.get("dropout", 0.1),
        A=A).to(device)

    module_df, total_params_M = summarize_model(net)
    print_model_summary(net)

    if cfg.get("optim","adamw").lower()=="adamw":
        optimizer = optim.AdamW(net.parameters(), lr=cfg["lr"],
                                weight_decay=cfg.get("wd",0.0))
    else:
        optimizer = optim.Adagrad(net.parameters(), lr=cfg["lr"])

    metrics_history = []

    scheduler, warmup = None, 0
    if "scheduler" in cfg:
        sc = cfg["scheduler"]
        if sc["name"].lower()=="cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg["epochs"]-sc["warmup_epochs"],
                eta_min=float(sc["min_lr"])
            )
            warmup = sc["warmup_epochs"]

    # DATA LAODING
    if cfg.get("class_weights", False):
        sampler, class_weights, dataset_tr = compute_class_weights(cfg)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        loader_tr = DataLoader(dataset_tr, batch_size=cfg['batch_size'],
                               sampler=sampler, num_workers=4,
                              drop_last=True, pin_memory=True)
        ce = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loader_tr = get_loader("train", cfg['batch_size'], **cfg["data"])
        ce = nn.CrossEntropyLoss()


    loader_va = get_loader("val",   64, **cfg["data"])
    mse = nn.MSELoss()
    start_time = time.time()

    print(f"Training {cfg['arch']} on {device} with batch size {cfg['batch_size']} for {cfg['epochs']} epochs")

    es = cfg.get("early_stopping",{})
    patience  = es.get("patience",25)
    min_delta = es.get("delta", es.get("min_delta",0.003))
    delta_f1  = es.get("delta_f1", 0.005)

    best_val   = float("inf")
    best_f1    = -float("inf")
    best_epoch = 0

    for epoch in range(1, cfg["epochs"]+1):
        tr = run_epoch(net, loader_tr, optimizer, cfg, ce, mse, device, train=True)
        va = run_epoch(net, loader_va, optimizer, cfg, ce, mse, device, train=False)
        lr = optimizer.param_groups[0]['lr']
        print(f"[{epoch:03d}]  tr_mpjpe={tr['mpjpe']:.3f}  va_mpjpe={va['mpjpe']:.3f}  "
              f"tr_f1={tr['f1']:.3f}  va_f1={va['f1']:.3f}  LR={lr:.2e}")
        improved_mpjpe = va['mpjpe'] + min_delta < best_val
        improved_f1    = (va["f1"] - best_f1)  > delta_f1
        if improved_mpjpe or improved_f1:
            best_val, best_f1, best_epoch = va["mpjpe"], va["f1"], epoch
            print(f" → saving model (mpjpe={va['mpjpe']:.3f}, f1={va['f1']:.3f})")
            torch.save(net.state_dict(), f"weights/weights_{cfg['arch']}.pt")
        elif epoch - best_epoch >= patience:
            print(f"Early stop at epoch {epoch} (no improve for {patience} epochs)")
            break

        if scheduler:
            if epoch <= warmup:
                lr_scale = epoch / warmup
                for g in optimizer.param_groups:
                    g['lr'] = lr_scale * cfg["lr"]
            else:
                scheduler.step()

        metrics_history.append({
            'epoch': int(epoch),
            'tr_mpjpe': tr['mpjpe'],
            'va_mpjpe': va['mpjpe'],
            'tr_ade': tr['ade'],
            'va_ade': va['ade'],
            'tr_fde': tr['fde'],
            'va_fde': va['fde'],
            'tr_f1': tr['f1'],
            'va_f1': va['f1'],
        })

    net.load_state_dict(torch.load(f"weights/weights_{cfg['arch']}.pt"))
    loader_te = get_loader("test", 64, **cfg["data"])
    te = run_epoch(net, loader_te, optimizer, cfg, ce, mse, device, False)
    row = f"{cfg['arch']},{te['mpjpe']:.3f},{te['ade']:.3f}, {te['fde']:.3f},{te['f1']:.3f}"
    print("CSV→", row)
    save_md_and_plot(metrics_history, cfg['arch'], cfg['batch_size'], total_params_M, module_df)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training time: {int(hours)}:{int(minutes):02}:{int(seconds):02}")
    torch.save(net.state_dict(), f"weights/weights_{cfg['arch']}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bilstm")
    parser.add_argument("--config", default="cfg/config.yaml")
    args = parser.parse_args()
    print(f'NOW RUNNING {args.model} with config {args.config}')
    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    main(yaml_cfg[args.model])
