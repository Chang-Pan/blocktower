import argparse
import os

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R


def get_cube_vertices_from_euler(pos, size, euler_xyz):
    """Compute cube vertices from center/size/euler angles (xyz)."""
    lx, ly, lz = size / 2.0
    v = np.array(
        [
            [-lx, -ly, -lz],
            [-lx, -ly, lz],
            [-lx, ly, -lz],
            [-lx, ly, lz],
            [lx, -ly, -lz],
            [lx, -ly, lz],
            [lx, ly, -lz],
            [lx, ly, lz],
        ]
    )
    quat = R.from_euler("xyz", euler_xyz, degrees=False).as_quat()
    v_rot = R.from_quat(quat).apply(v)
    return v_rot + pos


def get_cube_faces(vertices):
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]],
        [vertices[4], vertices[5], vertices[7], vertices[6]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[2], vertices[6], vertices[4]],
        [vertices[1], vertices[3], vertices[7], vertices[5]],
    ]
    return faces


def pick_top_dynamic_indices(frame0, dynamic_idx):
    block_indices = []
    for i in range(frame0.shape[0]):
        if frame0[i, dynamic_idx] == 1:
            block_indices.append((i, frame0[i, 2]))
    block_indices.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in block_indices[:5]]


def draw_one_axes(ax, traj, frame, title, colors, top_5_indices):
    pos_slice = slice(0, 3)
    euler_slice = slice(3, 6)
    size_slice = slice(6, 9)
    dynamic_idx = 9

    ax.clear()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 10])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Height)")
    ax.set_title(f"{title} - Step: {frame}")

    current = traj[frame]
    num_objs = current.shape[0]

    for i in range(num_objs):
        pos = current[i, pos_slice]
        euler_xyz = current[i, euler_slice]
        size = current[i, size_slice]

        if i == 0:
            color = "gray"
            alpha = 0.3
        else:
            color = colors[i % 10] if i in top_5_indices else "skyblue"
            alpha = 0.3

        vertices = get_cube_vertices_from_euler(pos, size, euler_xyz)
        faces = get_cube_faces(vertices)
        poly = art3d.Poly3DCollection(
            faces, facecolors=color, linewidths=0.2, edgecolors="black", alpha=alpha
        )
        poly.set_zorder(1)
        ax.add_collection3d(poly)

    for i in top_5_indices:
        history = traj[: frame + 1, i, pos_slice]
        ax.plot(
            history[:, 0],
            history[:, 1],
            history[:, 2],
            color=colors[i % 10],
            linewidth=3,
            alpha=1.0,
            zorder=10,
        )

    for i in range(num_objs):
        if current[i, dynamic_idx] == 1:
            pos = current[i, pos_slice]
            ax.scatter3D(
                pos[0],
                pos[1],
                pos[2],
                color="black",
                s=5,
                edgecolors="black",
                linewidth=0.8,
                depthshade=False,
                alpha=1.0,
                zorder=100,
            )


def visualize_comparison(pred_traj, true_traj, save_path, fps=25):
    num_frames = pred_traj.shape[0]
    num_objs = pred_traj.shape[1]

    fig = plt.figure(figsize=(20, 10))
    ax_true = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, num_objs))
    top_5_indices = pick_top_dynamic_indices(true_traj[0], dynamic_idx=9)

    def update(frame):
        draw_one_axes(ax_true, true_traj, frame, "Ground Truth", colors, top_5_indices)
        draw_one_axes(ax_pred, pred_traj, frame, "Prediction", colors, top_5_indices)
        return ax_true, ax_pred

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps)
    anim.save(save_path, writer="pillow")
    plt.close()
    print(f"Comparison animation saved to {save_path}")


def visualize_single_trajectory(traj_data, save_path, title="Trajectory", fps=25):
    num_frames = traj_data.shape[0]
    num_objs = traj_data.shape[1]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, num_objs))
    top_5_indices = pick_top_dynamic_indices(traj_data[0], dynamic_idx=9)

    def update(frame):
        draw_one_axes(ax, traj_data, frame, title, colors, top_5_indices)
        return (ax,)

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps)
    anim.save(save_path, writer="pillow")
    plt.close()
    print(f"Animation saved to {save_path}")


def visualize_from_npz(results_path, output_dir, fps=25):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = np.load(results_path, allow_pickle=True)
    stable_scenes = data["stable_scenes"]
    unstable_scenes = data["unstable_scenes"]

    print(f"Loaded validation data from: {results_path}")
    print(f"  Stable scenes: {len(stable_scenes)}")
    print(f"  Unstable scenes: {len(unstable_scenes)}")
    print("\nGenerating Euler visualizations...")

    for idx, scene in enumerate(stable_scenes):
        scene_name = scene["name"].replace(".npy", "")
        pred_traj = scene["pred"]
        true_traj = scene["true"]

        print(f"  Processing stable scene {idx + 1}/{len(stable_scenes)}: {scene_name}")
        comparison_path = os.path.join(output_dir, f"stable_{idx}_{scene_name}_comparison.gif")
        visualize_comparison(pred_traj, true_traj, comparison_path, fps=fps)

        pred_path = os.path.join(output_dir, f"stable_{idx}_{scene_name}_pred.gif")
        true_path = os.path.join(output_dir, f"stable_{idx}_{scene_name}_true.gif")
        visualize_single_trajectory(pred_traj, pred_path, title="Prediction", fps=fps)
        visualize_single_trajectory(true_traj, true_path, title="Ground Truth", fps=fps)

    for idx, scene in enumerate(unstable_scenes):
        scene_name = scene["name"].replace(".npy", "")
        pred_traj = scene["pred"]
        true_traj = scene["true"]

        print(f"  Processing unstable scene {idx + 1}/{len(unstable_scenes)}: {scene_name}")
        comparison_path = os.path.join(output_dir, f"unstable_{idx}_{scene_name}_comparison.gif")
        visualize_comparison(pred_traj, true_traj, comparison_path, fps=fps)

        pred_path = os.path.join(output_dir, f"unstable_{idx}_{scene_name}_pred.gif")
        true_path = os.path.join(output_dir, f"unstable_{idx}_{scene_name}_true.gif")
        visualize_single_trajectory(pred_traj, pred_path, title="Prediction", fps=fps)
        visualize_single_trajectory(true_traj, true_path, title="Ground Truth", fps=fps)

    print(f"\nVisualization complete! Results saved to: {output_dir}")
    print(f"Generated {3 * (len(stable_scenes) + len(unstable_scenes))} animations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        required=False,
        default="exps/my_exp/validation_data/debug_vis_epoch1000.npz",
        help="Path to debug_vis_epoch*.npz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exps/my_exp/animations_euler",
        help="Output directory for animations",
    )
    parser.add_argument("--fps", type=int, default=25, help="Animation frame rate")
    args = parser.parse_args()

    visualize_from_npz(args.results_path, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
