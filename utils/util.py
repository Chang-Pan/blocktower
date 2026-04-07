import torch
import numpy as np
import random
import csv
import pandas as pd
import re
import matplotlib.pyplot as plt

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

    import os
    log_dir = os.path.dirname(log_file)
    output_file = os.path.join(log_dir, f'loss_curve_split_{steps}.png')
    print(f'Saving plot to {output_file}')
    plt.savefig(output_file)
    plt.close()