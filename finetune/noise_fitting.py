"""
Fit human choices by injecting noise into predicted trajectories
and evaluating a non-differentiable rule-based decision head via Monte-Carlo.

Inputs:
  1) pred_npy: npy file saved by your evaluate pipeline
     - expected format: np.load(..., allow_pickle=True) returns an iterable of dict-like items
       each item has keys: 'name' (filename), 'pred' (trajectory: [T, O, 17])
  2) subject_path: path to csv with columns:
       - stimuli_file: scene name (include ".npy")
       - correct_answer: (unused here, optional)
       - user_choice: human choice (stable/dark_gray/light_gray)

Output:
  - prints best sigma + metrics
  - saves per-scene probabilities and per-row loglik to --out_dir

Example:
  python fit_noise_rule_based.py \
    --pred_npy exps/my_exp/predictions.npy \
    --subject_path ../data/subject_2/raw_data \
    --sim_steps 120 \
    --mc 200 \
    --sigma_min 0.0 --sigma_max 1.0 --sigma_num 60 \
    --seed 42 \
    --out_dir exps/noise_fit
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import glob

# -----------------------------
# 1) Your rule-based decision head
# -----------------------------
def decision_from_traj_fullpred(full_pred: np.ndarray, sim_steps: int) -> str:
    """
    full_pred: [T, O, 17]
    sim_steps: use min(sim_steps, T-1) then final is pred_traj[-1]
    IMPORTANT: keeps your original logic including "if sim_steps < 40 then use frame 39"
    Returns: "stable" | "light_gray" | "dark_gray"
    """
    if full_pred.ndim != 3 or full_pred.shape[-1] != 17:
        raise ValueError(f"full_pred must be [T,O,17], got {full_pred.shape}")

    # mimic original
    sim_steps_eff = min(sim_steps, full_pred.shape[0] - 1)
    if sim_steps_eff <= 0:
        pred_traj = np.array([full_pred[:sim_steps_eff]])
    else:
        pred_traj = full_pred[:sim_steps_eff]  # [time, obj, 17]

    original_scene = pred_traj[0]          # [obj,17]
    final_scene = pred_traj[-1]            # [obj,17]

    # stability check by z displacement
    is_stable = True
    for i in range(pred_traj.shape[1]):
        original_position = original_scene[i][:3]

        if sim_steps_eff < 40:
            # original code uses scene['pred'][39]
            if full_pred.shape[0] > 39:
                final_position = full_pred[39][i][:3]
            else:
                # fallback if trajectory shorter than 40 frames
                final_position = final_scene[i][:3]
        else:
            final_position = final_scene[i][:3]

        z_displacement = np.abs(final_position[2] - original_position[2])
        if z_displacement >= 0.6:
            is_stable = False
            break

    if is_stable:
        return "stable"

    # left/right decision by counting x>0 as light_gray
    light_gray = 0
    for i in range(pred_traj.shape[1]):
        if final_scene[i][0] > 0:
            light_gray += 1

    if light_gray >= (pred_traj.shape[1]) // 2:
        return "light_gray"
    else:
        return "dark_gray"


# -----------------------------
# 2) Noise injection
# -----------------------------
def add_noise_to_frames(full_pred: np.ndarray, sim_steps: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Inject Gaussian noise to the position (x,y,z) of:
      - final frame used by decision head
      - plus frame 39 if sim_steps < 40 and it exists (because stability check uses it)
    We DO NOT change velocities/quats/sizes here; only positions.

    sigma is in the same units as your stored pred positions.
    """
    noisy = full_pred.copy()

    # which final frame is used?
    sim_steps_eff = min(sim_steps, full_pred.shape[0] - 1)
    if sim_steps_eff <= 0:
        final_t = 0
    else:
        final_t = sim_steps_eff - 1  # because pred_traj = full_pred[:sim_steps_eff]

    # always noise the final frame (for left/right vote)
    if sigma > 0:
        noise = rng.normal(loc=0.0, scale=sigma, size=noisy[final_t, :, 0:3].shape)
        noisy[final_t, :, 0:3] += noise

    # if sim_steps<40, stability check uses frame 39
    if sim_steps_eff < 40 and full_pred.shape[0] > 39 and sigma > 0:
        noise39 = rng.normal(loc=0.0, scale=sigma, size=noisy[39, :, 0:3].shape)
        noisy[39, :, 0:3] += noise39

    return noisy


def mc_choice_probs(full_pred: np.ndarray, sim_steps: int, sigma: float, mc: int, seed: int, laplace: float = 1e-3):
    """
    Monte Carlo estimate of p(choice | scene, sigma) using your rule head.
    Returns:
      probs dict: {"stable": p0, "dark_gray": p1, "light_gray": p2}
    """
    rng = np.random.default_rng(seed)
    counts = {"stable": 0, "dark_gray": 0, "light_gray": 0}

    for k in range(mc):
        noisy = add_noise_to_frames(full_pred, sim_steps, sigma, rng)
        d = decision_from_traj_fullpred(noisy, sim_steps)
        counts[d] += 1

    # Laplace smoothing to avoid log(0)
    total = mc + 3 * laplace
    probs = {k: (counts[k] + laplace) / total for k in counts}
    return probs, counts


# -----------------------------
# 3) Human CSV parsing + label mapping
# -----------------------------
LABELS = ["stable", "dark_gray", "light_gray"]

def normalize_scene_name(x: str) -> str:
    x = str(x)
    file_info = x.split('/')[-1].split('.')[0].split('_')
    return f"{file_info[0]}_{file_info[1]}_{file_info[2]}_{file_info[3]}_{file_info[4]}"

def normalize_choice(x):
    """
    Accept:
      - strings: stable/dark_gray/light_gray (case-insensitive)
      - strings: left/right (map to dark/light? you can change if needed)
      - ints: 0/1/2 where you define mapping
    Default mapping (you can edit if your coding differs):
      0 -> stable
      1 -> dark_gray
      2 -> light_gray
    """
    if pd.isna(x):
        return None

    s = str(x).strip().lower()
    if s in ["stable", "no_fall", "nofall", "none"]:
        return "stable"
    if s in ["dark_gray", "dark", "left", "left_more", "leftmore"]:
        return "dark_gray"
    if s in ["light_gray", "light", "right", "right_more", "rightmore"]:
        return "light_gray"

    # unknown
    return None


# -----------------------------
# 4) Evaluate NLL for a given sigma
# -----------------------------
def compute_nll_for_sigma(df: pd.DataFrame, preds_by_scene: dict, sim_steps: int, sigma: float,
                          mc: int, seed: int):
    """
    df: human rows (each row is one subject x one scene, with a user_choice)
    preds_by_scene: scene_name -> full_pred [T,O,17]
    Returns:
      nll (float),
      acc (float),
      per_scene_probs (dict scene -> probs dict),
      per_row_logp (np.array)
    """
    per_scene_probs = {}
    per_row_logp = np.full((len(df),), np.nan, dtype=float)
    correct = 0
    used = 0
    nll = 0.0

    # precompute probs per scene (cache)
    for scene_name in df['scene_name']:
        if scene_name not in preds_by_scene:
            continue
        probs, _counts = mc_choice_probs(preds_by_scene[scene_name], sim_steps, sigma, mc, seed=seed + hash(scene_name) % 10_000)
        per_scene_probs[scene_name] = probs

    for i, row in df.iterrows():
        scene = row["scene_name"]
        y = row["choice_norm"]
        if y is None:
            continue
        if scene not in per_scene_probs:
            continue

        p = per_scene_probs[scene].get(y, 1e-12)
        lp = np.log(max(p, 1e-12))
        per_row_logp[i] = lp
        nll -= lp
        used += 1

        # MAP prediction accuracy
        probs = per_scene_probs[scene]
        pred = max(probs, key=lambda k: probs[k])
        if pred == y:
            correct += 1

    acc = correct / used if used > 0 else float("nan")
    nll = nll / used if used > 0 else float("nan")  # average NLL per trial
    return nll, acc, per_scene_probs, per_row_logp, used


# -----------------------------
# 5) Main: grid search sigma
# -----------------------------
def build_sigma_grid(sigma_min, sigma_max, sigma_num, grid_type="linear"):
    if sigma_num <= 1:
        return np.array([sigma_min], dtype=float)
    if grid_type == "log":
        # avoid log(0)
        lo = max(sigma_min, 1e-6)
        hi = max(sigma_max, lo * 1.01)
        return np.exp(np.linspace(np.log(lo), np.log(hi), sigma_num))
    return np.linspace(sigma_min, sigma_max, sigma_num)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_npy", type=str, required=True, help="npy from evaluate.py containing list of dicts with keys 'name','pred'")
    ap.add_argument("--subject_path", type=str, required=True, help="Path to subject(human) csv directory, including raw_data")
    ap.add_argument("--sim_steps", type=int, default=120)
    ap.add_argument("--mc", type=int, default=200, help="MC samples per scene to estimate choice probs")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--sigma_min", type=float, default=0.0)
    ap.add_argument("--sigma_max", type=float, default=1.0)
    ap.add_argument("--sigma_num", type=int, default=60)
    ap.add_argument("--grid_type", type=str, default="linear", choices=["linear", "log"])

    ap.add_argument("--out_dir", type=str, default="exps/noise_fit")
    ap.add_argument("--save_probs", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load predictions npy ----
    raw = np.load(args.pred_npy, allow_pickle=True)
    preds_by_scene = {}
    missing_keys = 0
    for item in raw:
        try:
            name = item["name"]
            pred = item["pred"]
        except Exception:
            missing_keys += 1
            continue
        scene_name = normalize_scene_name(name)
        preds_by_scene[scene_name] = pred

    if len(preds_by_scene) == 0:
        raise RuntimeError("No valid scenes found in pred_npy. Check format: item['name'], item['pred'].")

    print(f"[Load pred] scenes={len(preds_by_scene)} (skipped invalid items={missing_keys})")

    # ---- load human csv ----
    csv_files = sorted(glob.glob(os.path.join(args.subject_path, "*.csv")))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No csv files found under: {args.subject_path}")

    df_list = []
    for fp in csv_files:
        dfi = pd.read_csv(fp)
        df_list.append(dfi)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    required_cols = ["stimuli_file", "correct_answer", "user_choice"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Each csv must contain column '{c}'. Got columns: {list(df.columns)}")

    df = df.copy()
    df["scene_name"] = df["stimuli_file"].apply(normalize_scene_name)
    df["answer"] = df["correct_answer"]
    df["choice_norm"] = df["user_choice"].apply(normalize_choice)

    n_unknown = df["choice_norm"].isna().sum()
    print(f"[Load human] files={len(csv_files)} rows={len(df)} unknown_choice={n_unknown}")

    # filter rows whose scenes exist and choices known
    df_use = df[df["scene_name"].isin(preds_by_scene.keys()) & df["choice_norm"].notna()].reset_index(drop=True)
    print(f"[Align] usable rows={len(df_use)} unique_scenes={df_use['scene_name'].nunique()}")

    if len(df_use) == 0:
        raise RuntimeError("No usable human rows after alignment. Check stimuli_file naming and user_choice coding.")

    # ---- grid search sigma ----
    sigmas = build_sigma_grid(args.sigma_min, args.sigma_max, args.sigma_num, args.grid_type)
    results = []

    best = {"sigma": None, "nll": float("inf"), "acc": -1, "per_scene_probs": None, "per_row_logp": None, "used": 0}

    for si, sigma in enumerate(sigmas):
        nll, acc, per_scene_probs, per_row_logp, used = compute_nll_for_sigma(
            df_use, preds_by_scene, args.sim_steps, float(sigma), args.mc, seed=args.seed
        )
        results.append({"sigma": float(sigma), "nll": float(nll), "acc": float(acc), "used": int(used)})
        print(f"[{si+1:03d}/{len(sigmas)}] sigma={sigma:.6f}  avgNLL={nll:.4f}  acc={acc:.4f}  used={used}")

        if np.isfinite(nll) and nll < best["nll"]:
            best.update({
                "sigma": float(sigma),
                "nll": float(nll),
                "acc": float(acc),
                "per_scene_probs": per_scene_probs,
                "per_row_logp": per_row_logp,
                "used": int(used),
            })

    print("\n========== BEST ==========")
    print(f"best_sigma = {best['sigma']}")
    print(f"best_avgNLL = {best['nll']:.6f}")
    print(f"best_acc = {best['acc']:.6f}")
    print(f"used_rows = {best['used']}")

    # ---- save outputs ----
    out_summary = {
        "pred_npy": args.pred_npy,
        "subject_path": args.subject_path,
        "sim_steps": args.sim_steps,
        "mc": args.mc,
        "seed": args.seed,
        "grid": {
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "sigma_num": args.sigma_num,
            "grid_type": args.grid_type,
        },
        "best": {k: best[k] for k in ["sigma", "nll", "acc", "used"]},
        "all_results": results,
    }
    with open(os.path.join(args.out_dir, "noise_fit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out_summary, f, ensure_ascii=False, indent=2)

    # save per-row loglik
    df_out = df_use.copy()
    df_out["logp_under_best_sigma"] = best["per_row_logp"]
    df_out.to_csv(os.path.join(args.out_dir, "human_rows_with_logp.csv"), index=False)

    # save per-scene probs
    if args.save_probs and best["per_scene_probs"] is not None:
        rows = []
        for i, row in df.iterrows():
            scene = row["scene_name"]
            probs = best["per_scene_probs"][scene]
            answer = row["answer"]
            rows.append({
                "scene_name": scene,
                "p_stable": probs["stable"],
                "p_dark_gray": probs["dark_gray"],
                "p_light_gray": probs["light_gray"],
                "correct_answer": answer,
                "map_pred": max(probs, key=lambda k: probs[k]),
            })
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "per_scene_probs_best.csv"), index=False)

    print(f"\nSaved to: {args.out_dir}")
    print(" - noise_fit_summary.json")
    print(" - human_rows_with_logp.csv")
    print(" - per_scene_probs_best.csv (if enabled)")


if __name__ == "__main__":
    main()