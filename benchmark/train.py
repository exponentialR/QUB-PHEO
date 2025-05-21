import argparse, yaml, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from models.models import TwoHeadNet
from utils.metrics_utils import mpjpe_2d, ade_fde_2d, save_md_and_plot
from models.helper_utils import build_hand_adjacency, summarize_model, print_model_summary, set_seed, compute_class_weights, get_loader
import time
import torch


def run_epoch(nnet, loader, opt, cfg, mse, device, train=True):
    if train: nnet.train()
    else: nnet.eval()

    tots = {'mpjpe': 0, 'ade': 0, 'fde': 0, 'n': 0}

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

        if train: opt.zero_grad()

        yhat = nnet(x, use_intent_head=False)
        loss = mse(yhat, y_true)

        if train: loss.backward()
        nn.utils.clip_grad_norm_(nnet.parameters(), cfg.get('grad_clip', 1.0))
        opt.step()

        with torch.no_grad():
            b = x.size(0)
            tots["mpjpe"] += mpjpe_2d(yhat, y_true) * b
            ade, fde = ade_fde_2d(yhat, y_true)
            tots["ade"] += ade * b
            tots["fde"] += fde * b
            tots["n"] += b

    for k in ("mpjpe", "ade", "fde"):
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
    lr = float(cfg.get("lr", 0.001))
    wd = float(cfg.get("wd", 0.0))
    if cfg.get("optim","adamw").lower()=="adamw":
        optimizer = optim.AdamW(net.parameters(), lr=lr,
                                weight_decay=wd)
    else:
        optimizer = optim.Adagrad(net.parameters(), lr=lr)

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
        loader_tr = DataLoader(dataset_tr, batch_size=cfg['batch_size'],
                               sampler=sampler, num_workers=4,
                              drop_last=True, pin_memory=True)
    else:
        loader_tr = get_loader("train", cfg['batch_size'], **cfg["data"])


    loader_va = get_loader("val",   64, **cfg["data"])
    mse = nn.MSELoss()
    start_time = time.time()

    print(f"Training {cfg['arch']} on {device} with batch size {cfg['batch_size']} for {cfg['epochs']} epochs")

    es = cfg.get("early_stopping",{})
    patience  = es.get("patience",25)
    min_delta = es.get("delta", es.get("min_delta",0.003))

    best_val   = float("inf")
    best_epoch = 0

    for epoch in range(1, cfg["epochs"]+1):
        tr = run_epoch(net, loader_tr, optimizer, cfg, mse, device, train=True)
        va = run_epoch(net, loader_va, optimizer, cfg, mse, device, train=False)
        lr = optimizer.param_groups[0]['lr']
        print(f"[{epoch:03d}]  tr_mpjpe={tr['mpjpe']:.3f}  va_mpjpe={va['mpjpe']:.3f}  LR={lr:.2e}")
        improved_mpjpe = va['mpjpe'] + min_delta < best_val
        if improved_mpjpe:
            best_val, best_epoch = va["mpjpe"], epoch
            print(f" → saving model (mpjpe={va['mpjpe']:.3f})")
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
        })

    net.load_state_dict(torch.load(f"weights/weights_{cfg['arch']}.pt"))
    loader_te = get_loader("test", 64, **cfg["data"])
    te = run_epoch(net, loader_te, optimizer, cfg, mse, device, False)
    row = f"{cfg['arch']},{te['mpjpe']:.3f},{te['ade']:.3f}, {te['fde']:.3f}"
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
