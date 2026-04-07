import os
import time
import json
import csv
import argparse
import random
import importlib
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna

from utils.euler_blocktower_data_nff import DebugData, process_stacking_data_dynamic


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(args, device, hidden_dim=None, layer_num=None, step_size=None, dist_boundary=None):
    # 支持短名 euler_neural_simulator / 全名 models.euler_neural_simulator
    try:
        model_module = importlib.import_module(args.model_name)
    except ImportError:
        model_module = importlib.import_module(f"models.{args.model_name}")

    ForceFieldPredictor = model_module.ForceFieldPredictor
    ODEFunc = model_module.ODEFunc
    NeuralODEModel = model_module.NeuralODEModel

    hidden_dim = args.hidden_dim if hidden_dim is None else hidden_dim
    layer_num = args.layer_num if layer_num is None else layer_num
    step_size = args.step_size if step_size is None else step_size
    dist_boundary = args.dist_boundary if dist_boundary is None else dist_boundary

    force_predictor = ForceFieldPredictor(
        hidden_dim=hidden_dim,
        output_layer=layer_num,
        use_dist_mask=args.use_dist_mask,
        use_dist_input=args.use_dist_input,
        dist_boundary=dist_boundary
    )
    ode_func = ODEFunc(force_predictor, mass=1.0)
    model = NeuralODEModel(ode_func, use_adjoint=args.use_adjoint, step_size=step_size)
    model.to(device)
    return model


def normalize_batch(body_prop, vel):
    # body_prop: [B, T, O, 10], vel: [B, T, O, 3]
    # 与 euler_1scene_posnormed_train.py 一致：按首帧位置做全局尺度归一化
    pos_initial = body_prop[:, 0, :, 0:3]  # [B, O, 3]
    pos_flat = pos_initial.reshape(body_prop.size(0), -1)
    scene_scale = torch.max(torch.abs(pos_flat), dim=1)[0]
    scene_scale = torch.clamp(scene_scale, min=1.0)
    scale_view = scene_scale.view(-1, 1, 1, 1)

    body_prop = body_prop.clone()
    vel = vel.clone()

    body_prop[..., 0:3] /= scale_view   # position
    body_prop[..., 6:9] /= scale_view   # size(lx,ly,lz)
    vel /= scale_view                   # linear velocity

    return body_prop, vel, scene_scale


def _is_finite_tensor(x):
    return torch.isfinite(x).all().item()


def _raise_pruned_nonfinite(trial, where, epoch):
    raise optuna.TrialPruned(f"non-finite detected at {where}, epoch={epoch + 1}, trial={trial.number}")


def _has_nonfinite_grad(model):
    for p in model.parameters():
        if p.grad is not None and (not torch.isfinite(p.grad).all().item()):
            return True
    return False


def _has_nonfinite_param(model):
    for p in model.parameters():
        if not torch.isfinite(p).all().item():
            return True
    return False


@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device, trial=None, epoch=None):
    model.eval()
    val_loss = 0.0
    val_loss_pos = 0.0
    val_loss_euler = 0.0

    for _, body_prop, vel, ang_vel, _ in val_loader:
        body_prop = body_prop.to(device)
        vel = vel.to(device)
        ang_vel = ang_vel.to(device)

        body_prop, vel, scene_scale = normalize_batch(body_prop, vel)

        # 欧拉角版本监督 pos+euler
        true_traj = body_prop[..., 0:6].clone()

        z0 = torch.cat([
            body_prop[:, 0, :, :],
            vel[:, 0, :, :],
            ang_vel[:, 0, :, :]
        ], dim=-1)

        sim_steps = true_traj.shape[1]
        t = torch.linspace(0, (sim_steps - 1) / 25.0, steps=sim_steps, device=device).unsqueeze(0)

        try:
            pred_traj = model(z0, t, scene_scale=scene_scale)
        except Exception as e:
            msg = str(e).lower()
            if ("nan" in msg) or ("inf" in msg) or ("non-finite" in msg):
                if trial is not None and epoch is not None:
                    _raise_pruned_nonfinite(trial, "validate_forward_exception", epoch)
            raise

        if not _is_finite_tensor(pred_traj):
            if trial is not None and epoch is not None:
                _raise_pruned_nonfinite(trial, "validate_pred_traj", epoch)
            return float("inf"), float("inf"), float("inf")

        pred_pos = pred_traj[..., 0:3]
        pred_euler = pred_traj[..., 3:6]

        true_pos = true_traj[..., 0:3]
        true_euler = true_traj[..., 3:6]

        loss_pos = criterion(pred_pos, true_pos)
        loss_euler = criterion(pred_euler, true_euler)
        loss = loss_pos + loss_euler

        if (not _is_finite_tensor(loss_pos)) or (not _is_finite_tensor(loss_euler)) or (not _is_finite_tensor(loss)):
            if trial is not None and epoch is not None:
                _raise_pruned_nonfinite(trial, "validate_loss", epoch)
            return float("inf"), float("inf"), float("inf")

        val_loss += loss.item()
        val_loss_pos += loss_pos.item()
        val_loss_euler += loss_euler.item()

    num_batches = len(val_loader)
    val_loss /= num_batches
    val_loss_pos /= num_batches
    val_loss_euler /= num_batches

    model.train()
    return val_loss, val_loss_pos, val_loss_euler


def run_one_trial(trial, args, device):
    # 1) lr 与 eta_min 比率
    lr_start = trial.suggest_float("lr_start", args.lr_min, args.lr_max, log=True)
    lr_ratio = trial.suggest_float("lr_ratio", args.ratio_min, args.ratio_max, log=True)
    eta_min = max(lr_start * lr_ratio, 1e-8)

    # 2) 结构与求解器相关
    hidden_dim = trial.suggest_categorical("hidden_dim", args.hidden_dim_choices)
    layer_num = trial.suggest_categorical("layer_num", args.layer_num_choices)
    segment_len = trial.suggest_categorical("segment_len", args.segment_len_choices)
    step_size = trial.suggest_float("step_size", args.step_size_min, args.step_size_max, log=True)
    dist_boundary = trial.suggest_float("dist_boundary", args.dist_boundary_min, args.dist_boundary_max)

    # 3) 可选搜索 wd
    if args.tune_weight_decay:
        weight_decay = trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True)
    else:
        weight_decay = args.weight_decay

    set_seed(args.seed + trial.number)

    model = build_model(
        args,
        device,
        hidden_dim=hidden_dim,
        layer_num=layer_num,
        step_size=step_size,
        dist_boundary=dist_boundary
    )
    criterion = nn.MSELoss()

    dataset = DebugData(
        data_path=args.data_path,
        max_len=args.max_len,
        single_scene=args.single_scene,
        block_cnt=args.block_cnt,
        scene_type=args.scene_type
    )

    # 按你当前 debug 训练习惯：固定 batch=1
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr_start, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=eta_min
    )

    best_val = float("inf")
    last_reported = float("inf")
    best_epoch = -1
    best_state_dict = None

    trial_best_ckpt = ""
    if args.save_trial_ckpt:
        trial_dir = os.path.join(args.save_dir, "trial_ckpts", f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        trial_best_ckpt = os.path.join(trial_dir, "model_best.pt")

    for epoch in range(args.epochs):
        for _, body_prop, vel, ang_vel, _ in train_loader:
            body_prop = body_prop.to(device)
            vel = vel.to(device)
            ang_vel = ang_vel.to(device)

            body_prop, vel, scene_scale = normalize_batch(body_prop, vel)
            true_traj = body_prop[..., 0:6].clone()

            body_prop_s, vel_s, ang_vel_s, true_traj_s = process_stacking_data_dynamic(
                body_prop, true_traj, vel, ang_vel, SEGMENTS=segment_len
            )

            num_segments_per_sample = body_prop_s.shape[0] // body_prop.shape[0]
            scene_scale_expanded = scene_scale.repeat_interleave(num_segments_per_sample)

            z0 = torch.cat([
                body_prop_s[:, 0, :, :],
                vel_s[:, 0, :, :],
                ang_vel_s[:, 0, :, :]
            ], dim=-1)

            sim_steps = true_traj_s.shape[1]
            t = torch.linspace(0, (sim_steps - 1) / 25.0, steps=sim_steps, device=device).unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)

            try:
                pred_traj = model(z0, t, scene_scale=scene_scale_expanded)
            except Exception as e:
                msg = str(e).lower()
                if ("nan" in msg) or ("inf" in msg) or ("non-finite" in msg):
                    _raise_pruned_nonfinite(trial, "train_forward_exception", epoch)
                raise

            if not _is_finite_tensor(pred_traj):
                _raise_pruned_nonfinite(trial, "train_pred_traj", epoch)

            pred_pos = pred_traj[..., 0:3]
            pred_euler = pred_traj[..., 3:6]

            true_pos = true_traj_s[..., 0:3]
            true_euler = true_traj_s[..., 3:6]

            loss_pos = criterion(pred_pos, true_pos)
            loss_euler = criterion(pred_euler, true_euler)
            loss = loss_pos + loss_euler

            if (not _is_finite_tensor(loss_pos)) or (not _is_finite_tensor(loss_euler)) or (not _is_finite_tensor(loss)):
                _raise_pruned_nonfinite(trial, "train_loss", epoch)

            loss.backward()

            if _has_nonfinite_grad(model):
                _raise_pruned_nonfinite(trial, "train_grad", epoch)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if _has_nonfinite_param(model):
                _raise_pruned_nonfinite(trial, "train_param_after_step", epoch)

        scheduler.step()

        if ((epoch + 1) % args.val_interval == 0) or (epoch == args.epochs - 1):
            val_loss, val_loss_pos, val_loss_euler = validate_epoch(
                model, val_loader, criterion, device, trial=trial, epoch=epoch
            )
            last_reported = val_loss

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if ((epoch + 1) % args.report_interval == 0) or (epoch == args.epochs - 1):
                trial.report(val_loss, step=epoch + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    # trial 结束时最多保存一次最优 ckpt
    if args.save_trial_ckpt and (best_state_dict is not None):
        torch.save(
            {
                "state_dict": best_state_dict,
                "epoch": best_epoch,
                "val_loss": float(best_val),
                "trial_number": trial.number,
                "lr_start": float(lr_start),
                "lr_ratio": float(lr_ratio),
                "eta_min": float(eta_min),
                "weight_decay": float(weight_decay),
                "hidden_dim": int(hidden_dim),
                "layer_num": int(layer_num),
                "segment_len": int(segment_len),
                "step_size": float(step_size),
                "dist_boundary": float(dist_boundary),
                "args": vars(args),
            },
            trial_best_ckpt
        )

    trial.set_user_attr("best_epoch", int(best_epoch))
    trial.set_user_attr("best_ckpt_path", trial_best_ckpt if args.save_trial_ckpt else "")
    trial.set_user_attr("last_val", float(last_reported))
    trial.set_user_attr("best_val", float(best_val))
    trial.set_user_attr("eta_min", float(eta_min))
    trial.set_user_attr("lr_start", float(lr_start))
    trial.set_user_attr("weight_decay", float(weight_decay))
    trial.set_user_attr("hidden_dim", int(hidden_dim))
    trial.set_user_attr("layer_num", int(layer_num))
    trial.set_user_attr("segment_len", int(segment_len))
    trial.set_user_attr("step_size", float(step_size))
    trial.set_user_attr("dist_boundary", float(dist_boundary))
    trial.set_user_attr("params_million", count_parameters(model) / 1e6)

    return best_val


def save_trials(study, out_csv):
    fieldnames = [
        "number", "state", "value",
        "lr_start", "lr_ratio", "eta_min", "weight_decay",
        "hidden_dim", "layer_num", "segment_len", "step_size", "dist_boundary",
        "last_val", "best_val"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in study.trials:
            row = {
                "number": t.number,
                "state": str(t.state),
                "value": t.value if t.value is not None else "",
                "lr_start": t.params.get("lr_start", ""),
                "lr_ratio": t.params.get("lr_ratio", ""),
                "eta_min": t.user_attrs.get("eta_min", ""),
                "weight_decay": t.params.get("weight_decay", ""),
                "hidden_dim": t.params.get("hidden_dim", ""),
                "layer_num": t.params.get("layer_num", ""),
                "segment_len": t.params.get("segment_len", ""),
                "step_size": t.params.get("step_size", ""),
                "dist_boundary": t.params.get("dist_boundary", ""),
                "last_val": t.user_attrs.get("last_val", ""),
                "best_val": t.user_attrs.get("best_val", "")
            }
            writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()

    # 数据与输出
    p.add_argument("--data_path", type=str, default="/mnt/nfs_project_a/chang/data_euler/data_euler/blocktower")
    p.add_argument("--save_dir", type=str, default="exps/optuna_euler_1scene")
    p.add_argument("--model_name", type=str, default="euler_neural_simulator")

    # 基础训练
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n_trials", type=int, default=60)
    p.add_argument("--timeout_sec", type=int, default=0)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--report_interval", type=int, default=5)
    p.add_argument("--save_trial_ckpt", action="store_true", default=False)

    # 默认参数（当某些项不搜时可用）
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--layer_num", type=int, default=3)
    p.add_argument("--segment_len", type=int, default=15)
    p.add_argument("--step_size", type=float, default=1 / 200)
    p.add_argument("--dist_boundary", type=float, default=0.01)
    p.add_argument("--use_dist_mask", action="store_true", default=True)
    p.add_argument("--use_dist_input", action="store_true", default=True)
    p.add_argument("--use_adjoint", action="store_true", default=False)

    # 数据筛选（对齐你当前 debug 用法）
    p.add_argument("--scene_type", type=str, default="unstable", choices=["all", "stable", "unstable"])
    p.add_argument("--max_len", type=int, default=150)
    p.add_argument("--block_cnt", type=int, default=2)
    p.add_argument("--single_scene", action="store_true", default=True)

    # 学习率搜索空间
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--lr_max", type=float, default=1e-2)
    p.add_argument("--ratio_min", type=float, default=1e-3)
    p.add_argument("--ratio_max", type=float, default=3e-1)

    # wd 搜索（默认不开）
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--tune_weight_decay", action="store_true", default=False)
    p.add_argument("--wd_min", type=float, default=1e-7)
    p.add_argument("--wd_max", type=float, default=1e-4)

    # 联合搜索空间
    p.add_argument("--hidden_dim_choices", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--layer_num_choices", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--segment_len_choices", type=int, nargs="+", default=[10, 15, 20, 30])
    p.add_argument("--step_size_min", type=float, default=1 / 500)
    p.add_argument("--step_size_max", type=float, default=1 / 80)
    p.add_argument("--dist_boundary_min", type=float, default=0.0)
    p.add_argument("--dist_boundary_max", type=float, default=0.02)

    # Optuna
    p.add_argument("--study_name", type=str, default="optuna_euler_joint_search")
    p.add_argument("--storage", type=str, default="")
    p.add_argument("--n_startup_trials", type=int, default=10)
    p.add_argument("--n_warmup_steps", type=int, default=40)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.n_warmup_steps
    )

    if args.storage.strip():
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=args.storage,
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner
        )

    start = time.time()
    study.optimize(
        lambda trial: run_one_trial(trial, args, device),
        n_trials=args.n_trials,
        timeout=(args.timeout_sec if args.timeout_sec > 0 else None),
        gc_after_trial=True,
        show_progress_bar=False
    )
    elapsed = time.time() - start

    best = study.best_trial
    best_lr_start = best.params["lr_start"]
    best_ratio = best.params["lr_ratio"]
    best_eta_min = best.user_attrs.get("eta_min", best_lr_start * best_ratio)

    best_ckpt_src = best.user_attrs.get("best_ckpt_path", "")
    best_ckpt_dst = os.path.join(args.save_dir, "model_best_optuna.pt")
    if best_ckpt_src and os.path.exists(best_ckpt_src):
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        print(f"Best checkpoint copied to: {best_ckpt_dst}")
    elif args.save_trial_ckpt:
        print("Warning: best checkpoint path not found.")
    else:
        print("Per-trial checkpoint saving is disabled; skip copying best ckpt.")

    print("\n===== Optuna Done =====")
    print(f"Elapsed: {elapsed / 60:.2f} min")
    print(f"Best trial: #{best.number}")
    print(f"Best objective (min val loss): {best.value:.8f}")
    print(f"Best lr_start: {best_lr_start:.8e}")
    print(f"Best lr_ratio: {best_ratio:.8e}")
    print(f"Best eta_min: {best_eta_min:.8e}")
    print(f"Best weight_decay: {best.user_attrs.get('weight_decay', args.weight_decay)}")
    print(f"Best hidden_dim: {best.user_attrs.get('hidden_dim')}")
    print(f"Best layer_num: {best.user_attrs.get('layer_num')}")
    print(f"Best segment_len: {best.user_attrs.get('segment_len')}")
    print(f"Best step_size: {best.user_attrs.get('step_size')}")
    print(f"Best dist_boundary: {best.user_attrs.get('dist_boundary')}")

    best_json = {
        "best_trial": best.number,
        "best_value": best.value,
        "best_epoch": best.user_attrs.get("best_epoch", -1),
        "best_ckpt_path": best_ckpt_dst if os.path.exists(best_ckpt_dst) else "",
        "lr_start": float(best.params["lr_start"]),
        "lr_ratio": float(best.params["lr_ratio"]),
        "eta_min": float(best.user_attrs.get("eta_min", best.params["lr_start"] * best.params["lr_ratio"])),
        "weight_decay": float(best.user_attrs.get("weight_decay", args.weight_decay)),
        "hidden_dim": int(best.params["hidden_dim"]),
        "layer_num": int(best.params["layer_num"]),
        "segment_len": int(best.params["segment_len"]),
        "step_size": float(best.params["step_size"]),
        "dist_boundary": float(best.params["dist_boundary"]),
        "all_params": best.params,
        "all_user_attrs": best.user_attrs
    }

    best_json_path = os.path.join(args.save_dir, "best_params.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_json, f, ensure_ascii=False, indent=2)

    trials_csv_path = os.path.join(args.save_dir, "optuna_trials.csv")
    save_trials(study, trials_csv_path)

    print(f"Saved: {best_json_path}")
    print(f"Saved: {trials_csv_path}")


if __name__ == "__main__":
    main()