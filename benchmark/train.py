import argparse, yaml, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from datasets.dataset import QUBPHEOAerialDataset
from models.models import TwoHeadNet
from utils.metrics_utils import mpjpe_2d, ade_fde_2d, intent_f1, save_md_and_plot
import os
import time

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # make CUDA deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_loader(split, batch, **kwargs):
    """
    Returns a DataLoader for the given split.
    :param split: Split name (train, val, test)
    :param batch: Batch size
    :param kwargs: Additional arguments for DataLoader
    :return: DataLoader object
    """
    qpheo_dataset = QUBPHEOAerialDataset(split=split, **kwargs)
    return DataLoader(qpheo_dataset, batch_size=batch, shuffle=split=='train',
                      num_workers=4, drop_last=True, pin_memory=True)

def run_epoch(nnet, loader, opt, device, train=True):
    if train: nnet.train()
    else: nnet.eval()
    tots = {'mpjpe': 0, 'ade': 0, 'fde': 0, 'f1': 0, 'n': 0}
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    for batch in loader:
        x = batch['obs_h'].to(device)  # (B,60,84)
        y = batch['pred_h'].to(device)  # (B,60,84)
        lbl = batch['subtask'].to(device)  # (B,)
        if train:
            opt.zero_grad()
        yhat, logit = nnet(x)
        loss = mse(yhat, y) + ce(logit, lbl)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(nnet.parameters(), 1.0)
            opt.step()

        with torch.no_grad():
            tots["mpjpe"] += mpjpe_2d(yhat, y) * x.size(0)
            ade, fde = ade_fde_2d(yhat, y)
            tots["ade"] += ade * x.size(0);
            tots["fde"] += fde * x.size(0)
            tots["f1"] += intent_f1(logit, lbl) * x.size(0)
            tots["n"] += x.size(0)
    for k in ("mpjpe", "ade", "fde", "f1"):
        tots[k] /= tots["n"]
    return tots

def main(cfg):
    set_seed(cfg.get("seed", 42))

    metrics_history = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = TwoHeadNet(cfg["arch"]).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], amsgrad=True)
    loader_tr = get_loader("train", cfg['batch_size'], **cfg["data"])
    loader_va = get_loader("val",   64, **cfg["data"])
    start_time = time.time()

    print(f"Training {cfg['arch']} on {device} with batch size {cfg['batch_size']} for {cfg['epochs']} epochs")
    for epoch in range(1, cfg["epochs"]+1):
        tr = run_epoch(net, loader_tr, opt, device, True)
        va = run_epoch(net, loader_va, opt, device, False)
        print(f"[{epoch:02d}] "
              f"tr mpjpe {tr['mpjpe']:.3f}  "
              f"va mpjpe {va['mpjpe']:.3f}")
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

    # final evaluation on test
    loader_te = get_loader("test", 64, **cfg["data"])
    te = run_epoch(net, loader_te, opt, device, False)
    row = f"{cfg['arch']},{te['mpjpe']:.3f},{te['ade']:.3f}," \
          f"{te['fde']:.3f},{te['f1']:.3f}"
    print("CSV→", row)
    save_md_and_plot(metrics_history, cfg['arch'], cfg['batch_size'])
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

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    main(yaml_cfg[args.model])
