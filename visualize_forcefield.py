"""
visualize_forcefield.py  —  Neural Force Field Visualization for Block Tower

Generates publication-quality GIF animations showing learned force fields
overlaid on block tower simulations. Outputs three GIFs per scene:
  1. comparison (Ground Truth vs Prediction side-by-side)
  2. prediction only
  3. ground truth only

Each frame renders:
  • Semi-transparent 3D blocks with a scientific color palette
  • Force arrows (quiver) at each dynamic object's center of mass
  • Trajectory trails with gradient fade
  • Ground reference grid
  • Force magnitude colorbar

Usage:
  python visualize_forcefield.py \
    --results_path exps/xxx/validation_data/debug_vis_epoch1000.npz \
    --checkpoint  exps/xxx/model_best.pt \
    --scene_index 0 --scene_type stable \
    --start_frame 0 --end_frame 60 \
    --fps 10 --output_dir forcefield_vis

The model hyper-parameters (hidden_dim, layer_num, …) are read automatically
from the `args` dict stored inside the npz file.  You can override them via
command-line flags if needed.
"""

import argparse
import importlib
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

# ───────────────────────────── colour palette ─────────────────────────────
# Pastel scientific palette – more elegant than default tab10
BLOCK_COLORS = [
    "#4C72B0",  # steel blue
    "#DD8452",  # warm orange
    "#55A868",  # sage green
    "#C44E52",  # muted red
    "#8172B3",  # lavender purple
    "#937860",  # taupe
    "#DA8BC3",  # rose pink
    "#8C8C8C",  # grey
    "#CCB974",  # sand gold
    "#64B5CD",  # sky cyan
]
STATIC_COLOR = "#D5D5D5"
GROUND_GRID_COLOR = "#CCCCCC"
ARROW_CMAP = "coolwarm"
BG_COLOR = "#FAFAFA"

# ──────────────────────────── geometry helpers ────────────────────────────

def get_cube_vertices(pos, size, quat):
    """8 vertices of an oriented box given center, full-size, quaternion (xyzw)."""
    hx, hy, hz = size / 2.0
    v = np.array([
        [-hx, -hy, -hz], [-hx, -hy,  hz], [-hx,  hy, -hz], [-hx,  hy,  hz],
        [ hx, -hy, -hz], [ hx, -hy,  hz], [ hx,  hy, -hz], [ hx,  hy,  hz],
    ])
    v_rot = R.from_quat(quat).apply(v)   # scipy quat order is (x,y,z,w)
    return v_rot + pos


def get_cube_faces(vertices):
    """6 faces from 8 vertices."""
    idx = [
        [0, 1, 3, 2], [4, 5, 7, 6],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [0, 2, 6, 4], [1, 3, 7, 5],
    ]
    return [[vertices[i] for i in face] for face in idx]


def pick_top_dynamic_indices(frame0, n=5, dynamic_col=10):
    """Return indices of the *n* highest dynamic blocks (by initial z)."""
    candidates = []
    for i in range(frame0.shape[0]):
        if frame0[i, dynamic_col] == 1:
            candidates.append((i, frame0[i, 2]))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in candidates[:n]]


# ────────────────────────── force computation ─────────────────────────────

def compute_forces_for_frame(force_predictor, frame_data, device, scene_scale=None):
    """
    Run ForceFieldPredictor on a single frame and return per-object net force.

    Parameters
    ----------
    frame_data : ndarray [obj, 17]
        (x,y,z,qx,qy,qz,qw,lx,ly,lz,dynamic_mask, vx,vy,vz, wx,wy,wz)
    scene_scale : float or None

    Returns
    -------
    net_force : ndarray [obj, 3]   – net linear force (world frame)
    net_torque : ndarray [obj, 3]  – net torque (world frame)
    """
    obj_num = frame_data.shape[0]

    features = torch.tensor(frame_data[:, :11], dtype=torch.float32, device=device).unsqueeze(0)
    vel      = torch.tensor(frame_data[:, 11:14], dtype=torch.float32, device=device).unsqueeze(0)
    ang_vel  = torch.tensor(frame_data[:, 14:17], dtype=torch.float32, device=device).unsqueeze(0)

    sc = None
    if scene_scale is not None:
        sc = torch.tensor([scene_scale], dtype=torch.float32, device=device)

    with torch.no_grad():
        pairwise, ground = force_predictor(
            init_x=features, query_x=features,
            init_v=vel, query_v=vel,
            init_angular_v=ang_vel, query_angular_v=ang_vel,
            scene_scale=sc,
        )
    # pairwise: [1, obj, obj, 6]   ground: [1, obj, 6]

    # mask self-interaction
    mask = 1 - torch.eye(obj_num, device=device).unsqueeze(0).unsqueeze(-1)
    pairwise = pairwise * mask

    net_pair = pairwise.sum(dim=1).squeeze(0)   # [obj, 6]
    ground   = ground.squeeze(0)                 # [obj, 6]

    total = net_pair + ground                    # [obj, 6]

    # add gravity (z-axis = 2)
    gravity = -9.8
    if scene_scale is not None and scene_scale > 0:
        gravity = gravity / scene_scale
    total[:, 2] += gravity

    # mask by dynamic_mask
    dyn = torch.tensor(frame_data[:, 10:11], dtype=torch.float32, device=device)
    total = total * dyn

    return total[:, :3].cpu().numpy(), total[:, 3:6].cpu().numpy()


# ────────────────────────── drawing routines ──────────────────────────────

def _setup_ax(ax, title, frame_idx, xlim, ylim, zlim):
    """Configure a 3-D axes with clean scientific styling."""
    ax.clear()
    ax.set_facecolor(BG_COLOR)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X", fontsize=9, labelpad=2)
    ax.set_ylabel("Y", fontsize=9, labelpad=2)
    ax.set_zlabel("Z", fontsize=9, labelpad=2)
    ax.set_title(f"{title}  ·  frame {frame_idx}", fontsize=12, fontweight="bold",
                 pad=8, color="#333333")

    # reduce tick clutter
    ax.tick_params(axis="both", which="major", labelsize=7, pad=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    # subtle pane colours
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#E0E0E0")
    ax.yaxis.pane.set_edgecolor("#E0E0E0")
    ax.zaxis.pane.set_edgecolor("#E0E0E0")

    ax.view_init(elev=25, azim=135)


def _draw_ground_grid(ax, xlim, ylim, n=11):
    """Draw a faint ground-plane grid at z = 0."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    for x in xs:
        ax.plot([x, x], ylim, [0, 0], color=GROUND_GRID_COLOR, lw=0.4, alpha=0.5)
    for y in ys:
        ax.plot(xlim, [y, y], [0, 0], color=GROUND_GRID_COLOR, lw=0.4, alpha=0.5)


def _draw_blocks(ax, frame, top_indices, num_objs):
    """Render blocks as semi-transparent 3-D boxes."""
    for i in range(num_objs):
        pos  = frame[i, 0:3]
        quat = frame[i, 3:7]
        size = frame[i, 7:10]
        is_dynamic = frame[i, 10] > 0.5

        if not is_dynamic:
            color, alpha = STATIC_COLOR, 0.25
        elif i in top_indices:
            color = BLOCK_COLORS[i % len(BLOCK_COLORS)]
            alpha = 0.45
        else:
            color = "#B0D4F1"
            alpha = 0.30

        verts = get_cube_vertices(pos, size, quat)
        faces = get_cube_faces(verts)
        poly = art3d.Poly3DCollection(
            faces, facecolors=color, edgecolors="#555555",
            linewidths=0.3, alpha=alpha,
        )
        poly.set_zorder(1)
        ax.add_collection3d(poly)


def _draw_trails(ax, traj, frame_idx, top_indices, trail_len=30):
    """Trajectory trails with alpha gradient."""
    start = max(0, frame_idx - trail_len)
    for i in top_indices:
        hist = traj[start:frame_idx + 1, i, 0:3]
        if len(hist) < 2:
            continue
        color = BLOCK_COLORS[i % len(BLOCK_COLORS)]
        n = len(hist)
        for k in range(n - 1):
            alpha = 0.15 + 0.85 * (k / max(n - 1, 1))
            ax.plot(
                hist[k:k + 2, 0], hist[k:k + 2, 1], hist[k:k + 2, 2],
                color=color, linewidth=2.0, alpha=alpha, zorder=10,
            )


def _draw_forces(ax, frame, forces, force_scale, cmap, norm):
    """Draw force arrows at dynamic object centres."""
    num_objs = frame.shape[0]
    for i in range(num_objs):
        if frame[i, 10] < 0.5:
            continue
        f = forces[i]
        mag = np.linalg.norm(f)
        if mag < 1e-6:
            continue
        pos = frame[i, 0:3]
        color = cmap(norm(mag))
        ax.quiver(
            pos[0], pos[1], pos[2],
            f[0] * force_scale, f[1] * force_scale, f[2] * force_scale,
            color=color, arrow_length_ratio=0.25,
            linewidth=1.8, alpha=0.9, zorder=50,
        )


def _draw_centroids(ax, frame, num_objs):
    """Small black dots at dynamic block centres."""
    for i in range(num_objs):
        if frame[i, 10] > 0.5:
            pos = frame[i, 0:3]
            ax.scatter3D(
                pos[0], pos[1], pos[2],
                color="black", s=8, edgecolors="black",
                linewidth=0.5, depthshade=False, alpha=0.9, zorder=100,
            )


# ──────────────────── auto-detect axis limits & force scale ───────────────

def _compute_limits(traj, start, end):
    """Compute tight but padded axis limits from trajectory data."""
    sub = traj[start:end]
    pos = sub[:, :, 0:3]
    sizes = sub[:, :, 7:10]
    lo = (pos - sizes / 2).reshape(-1, 3).min(axis=0)
    hi = (pos + sizes / 2).reshape(-1, 3).max(axis=0)
    pad = (hi - lo) * 0.15 + 0.5
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])
    zlim = (max(lo[2] - pad[2], -0.5), hi[2] + pad[2])
    return xlim, ylim, zlim


def _estimate_force_scale(all_forces_list, target_arrow_len=0.8):
    """Choose a scale factor so median-magnitude arrows are a reasonable length."""
    mags = []
    for forces in all_forces_list:
        m = np.linalg.norm(forces, axis=-1)
        mags.append(m[m > 1e-6])
    if len(mags) == 0:
        return 1.0
    mags = np.concatenate(mags)
    if len(mags) == 0:
        return 1.0
    median_mag = np.median(mags)
    if median_mag < 1e-8:
        return 1.0
    return target_arrow_len / median_mag


# ────────────────────── high-level animation builders ─────────────────────

def build_single_animation(
    traj, forces_per_frame, save_path,
    title, start_frame, end_frame, fps,
    xlim, ylim, zlim, force_scale, force_norm,
    figsize=(10, 8), dpi=150,
):
    """Build and save a single-view GIF."""
    num_frames = end_frame - start_frame
    num_objs = traj.shape[1]
    top_indices = pick_top_dynamic_indices(traj[start_frame])

    cmap = plt.get_cmap(ARROW_CMAP)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")

    # static colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=force_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=20)
    cbar.set_label("Force magnitude", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    def update(idx):
        frame_idx = start_frame + idx
        _setup_ax(ax, title, frame_idx, xlim, ylim, zlim)
        _draw_ground_grid(ax, xlim, ylim)
        _draw_blocks(ax, traj[frame_idx], top_indices, num_objs)
        _draw_trails(ax, traj, frame_idx, top_indices)
        _draw_forces(ax, traj[frame_idx], forces_per_frame[idx], force_scale, cmap, force_norm)
        _draw_centroids(ax, traj[frame_idx], num_objs)
        return (ax,)

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps)
    anim.save(save_path, writer="pillow")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def build_comparison_animation(
    pred_traj, true_traj,
    pred_forces, true_forces,
    save_path, start_frame, end_frame, fps,
    xlim, ylim, zlim, force_scale, force_norm,
    dpi=150,
):
    """Build side-by-side Ground Truth | Prediction GIF."""
    num_frames = end_frame - start_frame
    num_objs = pred_traj.shape[1]
    top_indices = pick_top_dynamic_indices(true_traj[start_frame])

    cmap = plt.get_cmap(ARROW_CMAP)

    fig = plt.figure(figsize=(18, 8), dpi=dpi, facecolor=BG_COLOR)
    ax_true = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=force_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_true, ax_pred], shrink=0.55, pad=0.06, aspect=20)
    cbar.set_label("Force magnitude", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    def update(idx):
        frame_idx = start_frame + idx
        for ax, traj, forces, label in [
            (ax_true, true_traj, true_forces, "Ground Truth"),
            (ax_pred, pred_traj, pred_forces, "Prediction"),
        ]:
            _setup_ax(ax, label, frame_idx, xlim, ylim, zlim)
            _draw_ground_grid(ax, xlim, ylim)
            _draw_blocks(ax, traj[frame_idx], top_indices, num_objs)
            _draw_trails(ax, traj, frame_idx, top_indices)
            _draw_forces(ax, traj[frame_idx], forces[idx], force_scale, cmap, force_norm)
            _draw_centroids(ax, traj[frame_idx], num_objs)
        return ax_true, ax_pred

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps)
    anim.save(save_path, writer="pillow")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────── model loading ────────────────────────────────

def load_force_predictor(checkpoint_path, npz_args, device):
    """
    Reconstruct ForceFieldPredictor from saved args + checkpoint.
    Supports both quaternion (neural_simulator) and euler (euler_neural_simulator).
    """
    model_name = npz_args.get("model_name", "neural_simulator")
    hidden_dim = int(npz_args.get("hidden_dim", 256))
    layer_num  = int(npz_args.get("layer_num", 4))
    use_dist_mask  = bool(npz_args.get("use_dist_mask", True))
    use_dist_input = bool(npz_args.get("use_dist_input", True))
    dist_boundary  = float(npz_args.get("dist_boundary", 0.02))

    # Import the correct module (same fallback logic as training script)
    try:
        mod = importlib.import_module(model_name)
    except ImportError:
        try:
            mod = importlib.import_module(f"models.{model_name}")
        except ImportError as e:
            raise ImportError(
                f"Cannot import model '{model_name}'. "
                f"Make sure neural_simulator.py is on your PYTHONPATH."
            ) from e

    FFP = mod.ForceFieldPredictor
    predictor = FFP(
        hidden_dim=hidden_dim,
        output_layer=layer_num,
        use_dist_mask=use_dist_mask,
        dist_boundary=dist_boundary,
        use_dist_input=use_dist_input,
    )

    # Load state dict – extract only force_predictor weights
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    prefix = "ode_func.force_predictor."
    fp_state = {}
    for k, v in state.items():
        if k.startswith(prefix):
            fp_state[k[len(prefix):]] = v
    if len(fp_state) == 0:
        # maybe the checkpoint is already a force predictor
        fp_state = state

    predictor.load_state_dict(fp_state)
    predictor.to(device)
    predictor.eval()
    print(f"  Loaded ForceFieldPredictor from {checkpoint_path}  ({model_name})")
    return predictor


# ─────────────────────── scene scale estimation ───────────────────────────

def estimate_scene_scale(traj):
    """
    Estimate scene_scale the same way the training script does:
    max of abs(positions) and abs(sizes) across the first frame.
    """
    frame0 = traj[0]  # [obj, 17]
    pos_abs = np.abs(frame0[:, 0:3]).max()
    size_abs = np.abs(frame0[:, 7:10]).max()
    return max(pos_abs, size_abs, 1e-6)


# ───────────────────────────── main entry ─────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize learned force fields from Neural Force Field model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to debug_vis_epoch*.npz")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default="forcefield_vis",
                        help="Output directory for GIFs")
    parser.add_argument("--scene_index", type=int, default=0,
                        help="Which scene to visualise (0-indexed)")
    parser.add_argument("--scene_type", type=str, default="stable",
                        choices=["stable", "unstable"],
                        help="Scene category")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1,
                        help="End frame (exclusive). -1 = all frames")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no_force", action="store_true",
                        help="Skip force computation, draw blocks only")

    # model overrides (usually auto-read from npz)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--layer_num", type=int, default=None)
    parser.add_argument("--dist_boundary", type=float, default=None)
    parser.add_argument("--model_name", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load NPZ ──
    print("Loading data …")
    data = np.load(args.results_path, allow_pickle=True)
    scenes_key = f"{args.scene_type}_scenes"
    scenes = data[scenes_key]
    npz_args = data["args"].item() if "args" in data else {}

    if args.scene_index >= len(scenes):
        print(f"Error: scene_index={args.scene_index} but only {len(scenes)} "
              f"{args.scene_type} scenes available.")
        sys.exit(1)

    scene = scenes[args.scene_index]
    scene_name = scene["name"].replace(".npy", "")
    pred_traj = scene["pred"]   # [time, obj, 17]
    true_traj = scene["true"]   # [time, obj, 17]

    total_frames = pred_traj.shape[0]
    start = args.start_frame
    end = args.end_frame if args.end_frame > 0 else total_frames
    end = min(end, total_frames)
    print(f"  Scene: {scene_name}  |  frames {start}→{end}  ({end - start} frames)")
    print(f"  Objects: {pred_traj.shape[1]}  |  FPS: {args.fps}")

    # ── 2. Load model ──
    # merge overrides
    override_keys = ["hidden_dim", "layer_num", "dist_boundary", "model_name"]
    for k in override_keys:
        v = getattr(args, k, None)
        if v is not None:
            npz_args[k] = v

    if not args.no_force:
        predictor = load_force_predictor(args.checkpoint, npz_args, device)
    else:
        predictor = None

    # ── 3. Compute forces for every frame ──
    print("Computing forces …")
    pred_forces = []  # list of [obj, 3]
    true_forces = []

    # Estimate scene scale from true trajectory (same heuristic as training)
    scene_scale_val = estimate_scene_scale(true_traj)

    for f in range(start, end):
        if predictor is not None:
            pf, _ = compute_forces_for_frame(predictor, pred_traj[f], device, scene_scale=scene_scale_val)
            tf, _ = compute_forces_for_frame(predictor, true_traj[f], device, scene_scale=scene_scale_val)
        else:
            obj_n = pred_traj.shape[1]
            pf = np.zeros((obj_n, 3))
            tf = np.zeros((obj_n, 3))
        pred_forces.append(pf)
        true_forces.append(tf)
        if (f - start) % 20 == 0:
            print(f"    frame {f}/{end}")

    pred_forces = np.array(pred_forces)  # [num_frames, obj, 3]
    true_forces = np.array(true_forces)

    # ── 4. Determine shared visual parameters ──
    xlim, ylim, zlim = _compute_limits(
        np.concatenate([pred_traj, true_traj], axis=1), start, end
    )
    # recompute for each individually then merge
    xlim_p, ylim_p, zlim_p = _compute_limits(pred_traj, start, end)
    xlim_t, ylim_t, zlim_t = _compute_limits(true_traj, start, end)
    xlim = (min(xlim_p[0], xlim_t[0]), max(xlim_p[1], xlim_t[1]))
    ylim = (min(ylim_p[0], ylim_t[0]), max(ylim_p[1], ylim_t[1]))
    zlim = (min(zlim_p[0], zlim_t[0]), max(zlim_p[1], zlim_t[1]))

    force_scale = _estimate_force_scale([pred_forces, true_forces])
    all_mags = np.concatenate([
        np.linalg.norm(pred_forces, axis=-1).ravel(),
        np.linalg.norm(true_forces, axis=-1).ravel(),
    ])
    all_mags = all_mags[all_mags > 1e-6]
    if len(all_mags) > 0:
        vmax = np.percentile(all_mags, 95)
    else:
        vmax = 1.0
    force_norm = mcolors.Normalize(vmin=0, vmax=max(vmax, 1e-6))

    print(f"  Force scale: {force_scale:.4f}  |  Norm vmax: {vmax:.4f}")

    # ── 5. Generate GIFs ──
    tag = f"{args.scene_type}_{args.scene_index}_{scene_name}_f{start}-{end}"

    print("\nGenerating comparison GIF …")
    build_comparison_animation(
        pred_traj, true_traj, pred_forces, true_forces,
        save_path=os.path.join(args.output_dir, f"{tag}_comparison.gif"),
        start_frame=start, end_frame=end, fps=args.fps,
        xlim=xlim, ylim=ylim, zlim=zlim,
        force_scale=force_scale, force_norm=force_norm,
        dpi=args.dpi,
    )

    print("Generating prediction GIF …")
    build_single_animation(
        pred_traj, pred_forces,
        save_path=os.path.join(args.output_dir, f"{tag}_pred.gif"),
        title="Prediction", start_frame=start, end_frame=end, fps=args.fps,
        xlim=xlim, ylim=ylim, zlim=zlim,
        force_scale=force_scale, force_norm=force_norm,
        dpi=args.dpi,
    )

    print("Generating ground truth GIF …")
    build_single_animation(
        true_traj, true_forces,
        save_path=os.path.join(args.output_dir, f"{tag}_true.gif"),
        title="Ground Truth", start_frame=start, end_frame=end, fps=args.fps,
        xlim=xlim, ylim=ylim, zlim=zlim,
        force_scale=force_scale, force_norm=force_norm,
        dpi=args.dpi,
    )

    print(f"\n✅ Done!  {3} GIFs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
