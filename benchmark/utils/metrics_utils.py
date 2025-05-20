import time

import torch
import os
import matplotlib.pyplot as plt
import pandas as pd


def mpjpe_2d(pred, gt):
    """
    Mean Per Joint Position Error (MPJPE) in 2D.
    :param pred: Predicted 2D joint positions (N, J, 2)
    :param gt: Ground truth 2D joint positions (N, J, 2)
    :return: MPJPE value
    """
    assert pred.shape == gt.shape, "Predicted and ground truth shapes must match"
    return (pred -gt).norm(dim=-1).mean().item()

def ade_fde_2d(pred, gt):   # (B, T, 84) → (B, T, 2)
    pred_c = pred.view(*pred.shape[:-1], 42, 2).mean(dim=-2)
    gt_c   = gt.view(*gt.shape[:-1], 42, 2).mean(dim=-2)
    ade = (pred_c - gt_c).norm(dim=-1).mean().item()
    fde = (pred_c[:, -1] - gt_c[:, -1]).norm(dim=-1).mean().item()
    return ade, fde

def intent_f1(logits, labels):
    pred = logits.argmax(-1)
    tp = ((pred == labels) & (labels != -1)).sum().item()
    p  = max(pred.numel(), 1)
    r  = max((labels != -1).sum().item(), 1)
    if p + r == 0: return 0.
    return 2 * tp / (p + r)

def save_md_and_plot(metrics_history, arch, batch_size, total_params_M, module_df):
    """
    metrics_history: list of dicts, each with keys
        'epoch', 'tr_mpjpe', 'va_mpjpe', 'tr_ade', 'va_ade', etc.
    arch: model name (string) for filenames/headers
    """
    df = pd.DataFrame(metrics_history)


    md_path = f"results_{arch}.md"
    is_new = not os.path.exists(md_path)
    mode = 'w' if is_new else 'a'
    with open(md_path, mode) as f:
        if is_new:
            f.write(f"# {arch.upper()} Performance\n\n")
            f.write(f"- **Total params:** {total_params_M:.2f} M\n\n")
            f.write("## Model parameter breakdown\n\n")
            f.write(module_df.to_markdown(index=False, floatfmt=(".0f", ".3f")))
            f.write("\n\n")

        f.write(f"## Run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- **Train batch size:** {batch_size}\n")
        f.write("### Epoch-wise metrics\n\n")
        # df['epoch'] = df['epoch'].astype(int)
        f.write(df.to_markdown(index=False, floatfmt=[ ".0f" ] + [".3f"] * (df.shape[1]-1)))

        # f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n")
        img_name = f"plots/{time.strftime('%Y%m%d_%H%M%S')}_{arch}_learning_curves.png"
        f.write(f"## Learning curves\n\n")
        f.write(f"![Performance curves]({img_name})\n")

    plt.figure(figsize=(6,4))
    plt.plot(df['epoch'], df['tr_mpjpe'], label='Train MPJPE')
    plt.plot(df['epoch'], df['va_mpjpe'], label='Val MPJPE')
    plt.xlabel('Epoch')
    plt.ylabel('MPJPE (2D)')
    plt.title(f'{arch.upper()} MPJPE over Epochs')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(img_name, dpi=400)
    plt.close()

