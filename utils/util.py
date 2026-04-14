import torch
import numpy as np
import random
import csv
import pandas as pd
import re
import matplotlib.pyplot as plt
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_tensor_as_csv(tensor, file_path):
    
    newtensor = tensor.cpu()
    if len(tensor.shape) > 2:
        newtensor = newtensor.reshape(-1, tensor.shape[-1])

    newtensor = newtensor.detach().numpy()
    if not isinstance(tensor, np.ndarray):
        newtensor = np.array(newtensor)
    
    
    df = pd.DataFrame(newtensor)
    
    df.to_csv(file_path, index=False, header=False)

def calculate_distribution(data):

    logs = np.log10(np.abs(data))
    bins = np.floor(logs)
    bin_counts = {}
    for bin in bins:
        if bin in bin_counts:
            bin_counts[bin] += 1
        else:
            bin_counts[bin] = 1
    
    total_count = len(data)
    bin_counts = dict(sorted(bin_counts.items(), key=lambda x: x[0]))
    for bin, count in bin_counts.items():
        print(f'10^{bin} - 10^{bin+1}: {count/total_count:.4f}')


def vis_losscurve(steps, log_file, eps=1e-8):
    train_epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []

    # 支持新日志格式:
    # [TRAIN] Epoch [e/E] Loss: ..., MSE: ..., Residual Loss: ..., Residual Residual Loss: ...
    # [VAL]   Epoch [e/E] Loss: ..., MSE: ..., Residual Loss: ..., Residual Residual Loss: ...
    log_pattern = re.compile(
        r"\[(TRAIN|VAL)\]\s*Epoch\s*\[(\d+)/\d+\].*Loss:\s*([\d\.eE\-]+)\s*,\s*MSE:\s*([\d\.eE\-]+)\s*,\s*Residual\s*Loss:\s*([\d\.eE\-]+)\s*,\s*Residual\s*Residual\s*Loss:\s*([\d\.eE\-]+)"
    )

    with open(log_file, 'r', encoding='utf-8') as file:
        for line in file:
            match = log_pattern.search(line)
            if not match:
                continue

            split_name = match.group(1)  # TRAIN or VAL
            epoch = int(match.group(2))
            loss = float(match.group(3))

            if split_name == "TRAIN":
                train_epochs.append(epoch)
                train_losses.append(loss)
            else:
                val_epochs.append(epoch)
                val_losses.append(loss)

    if len(train_losses) == 0 and len(val_losses) == 0:
        print("No TRAIN/VAL logs matched. Please check logging format.")
        return

    # 防止 log(0)
    train_losses_log = np.log(np.array(train_losses, dtype=np.float64) + eps) if len(train_losses) > 0 else np.array([])
    val_losses_log = np.log(np.array(val_losses, dtype=np.float64) + eps) if len(val_losses) > 0 else np.array([])

    # 分开画: 上图 train, 下图 val
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    if len(train_losses) > 0:
        axes[0].plot(train_epochs, train_losses_log, label='Train Loss (log)', color='tab:blue')
        axes[0].set_title(f'Training Loss Curve for {steps} steps (log(loss+eps), eps={eps})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Log Loss')
        axes[0].grid(True)
        axes[0].legend()
    else:
        axes[0].set_title('Training Loss Curve (No Data)')
        axes[0].grid(True)

    if len(val_losses) > 0:
        axes[1].plot(val_epochs, val_losses_log, label='Validation Loss (log)', color='tab:orange')
        axes[1].set_title(f'Validation Loss Curve for {steps} steps (log(loss+eps), eps={eps})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Log Loss')
        axes[1].grid(True)
        axes[1].legend()
    else:
        axes[1].set_title('Validation Loss Curve (No Data)')
        axes[1].grid(True)

    plt.tight_layout()

    log_dir = os.path.dirname(log_file)
    output_file = os.path.join(log_dir, f'loss_curve_split_{steps}.png')
    print(f'Saving plot to {output_file}')
    plt.savefig(output_file)
    plt.close()


def vis_lrcurve_from_values(lr_values, output_file, epochs=None, title='Learning Rate Curve'):
    """
    Plot learning rate curve from a list/array of LR values.

    Args:
        lr_values: list/np.ndarray of LR values
        output_file: output image path
        epochs: optional list/np.ndarray of epoch indices (1-based recommended)
        title: figure title
    """
    if lr_values is None or len(lr_values) == 0:
        print('No lr_values provided. Skip plotting.')
        return

    lr_values = np.array(lr_values, dtype=np.float64)
    if epochs is None:
        epochs = np.arange(1, len(lr_values) + 1)
    else:
        epochs = np.array(epochs, dtype=np.int64)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, lr_values, color='tab:green', linewidth=1.8)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f'Saving LR curve to {output_file}')
    plt.savefig(output_file)
    plt.close()


def vis_lrcurve(log_file, output_file=None):
    """
    Parse and plot learning rate curve from log file.

    Supported line examples:
    - [LR] Epoch [12/1000] LR: 0.000300
    - Epoch [12/1000] ... LR: 3.0e-4
    - lr=0.0003

    Priority is epoch-aware matches; if no epoch is found, falls back to order index.
    """
    epoch_lr_pairs = []
    lr_only = []

    # Epoch-aware patterns
    p1 = re.compile(r"Epoch\s*\[(\d+)\s*/\s*\d+\].*?LR\s*[:=]\s*([\d\.eE\-\+]+)", re.IGNORECASE)
    p2 = re.compile(r"\[LR\]\s*Epoch\s*\[(\d+)\s*/\s*\d+\]\s*LR\s*[:=]\s*([\d\.eE\-\+]+)", re.IGNORECASE)
    # LR-only pattern fallback
    p3 = re.compile(r"\blr\b\s*[:=]\s*([\d\.eE\-\+]+)", re.IGNORECASE)

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            m = p1.search(line) or p2.search(line)
            if m:
                epoch_lr_pairs.append((int(m.group(1)), float(m.group(2))))
                continue

            m3 = p3.search(line)
            if m3:
                lr_only.append(float(m3.group(1)))

    if len(epoch_lr_pairs) == 0 and len(lr_only) == 0:
        print('No LR entries matched in log. Please add LR logging first.')
        return

    if output_file is None:
        log_dir = os.path.dirname(log_file)
        output_file = os.path.join(log_dir, 'lr_curve.png')

    if len(epoch_lr_pairs) > 0:
        # If duplicates exist for same epoch, keep the last one.
        epoch_to_lr = {}
        for ep, lr in epoch_lr_pairs:
            epoch_to_lr[ep] = lr
        epochs = sorted(epoch_to_lr.keys())
        lrs = [epoch_to_lr[e] for e in epochs]
        vis_lrcurve_from_values(lrs, output_file, epochs=epochs, title='Learning Rate Curve (from log epochs)')
        return

    # Fallback: no epoch info, plot by occurrence index.
    vis_lrcurve_from_values(lr_only, output_file, title='Learning Rate Curve (from log order)')