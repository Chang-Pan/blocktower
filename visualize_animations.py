import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.spatial.transform import Rotation as R
import os
import argparse

def get_cube_vertices(pos, size, quat):
    """
    根据中心点、尺寸和四元数计算立方体的8个顶点
    """
    # 1. 定义单位立方体的顶点 (中心在原点)
    _lx, _ly, _lz = size / 2.0
    v = np.array([
        [-_lx, -_ly, -_lz], [-_lx, -_ly, _lz], [-_lx, _ly, -_lz], [-_lx, _ly, _lz],
        [_lx, -_ly, -_lz], [_lx, -_ly, _lz], [_lx, _ly, -_lz], [_lx, _ly, _lz]
    ])
    
    # 2. 应用四元数旋转
    r = R.from_quat(quat) # quat顺序是 [x, y, z, w]
    v_rot = r.apply(v)
    
    # 3. 移动到实际位置
    v_final = v_rot + pos
    return v_final

def get_cube_faces(vertices):
    """根据8个顶点定义6个面"""
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]], # Back
        [vertices[4], vertices[5], vertices[7], vertices[6]], # Front
        [vertices[0], vertices[1], vertices[5], vertices[4]], # Bottom
        [vertices[2], vertices[3], vertices[7], vertices[6]], # Top
        [vertices[0], vertices[2], vertices[6], vertices[4]], # Left
        [vertices[1], vertices[3], vertices[7], vertices[5]]  # Right
    ]
    return faces

def visualize_comparison(pred_traj, true_traj, save_path, fps=25):
    """
    并排对比显示预测轨迹和真实轨迹
    
    Args:
        pred_traj: [time, obj_num, 17] 预测轨迹
        true_traj: [time, obj_num, 17] 真实轨迹
        save_path: 保存路径
        fps: 帧率
    """
    num_frames = pred_traj.shape[0]
    num_objs = pred_traj.shape[1]
    
    fig = plt.figure(figsize=(20, 10))
    ax_true = fig.add_subplot(121, projection='3d')
    ax_pred = fig.add_subplot(122, projection='3d')
    
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, num_objs))

    # 识别需要显示轨迹线的物体 (初始高度最高的5个动态物体)
    initial_frame = true_traj[0]
    block_indices = []
    for i in range(num_objs):
        if initial_frame[i, 10] == 1: # dynamic_mask=1 的是积木
            block_indices.append((i, initial_frame[i, 2])) # (index, z_height)
    
    # 按初始 Z 高度降序排列，取前5个
    block_indices.sort(key=lambda x: x[1], reverse=True)
    top_5_indices = [idx for idx, h in block_indices[:5]]
    
    def update(frame):
        ax_true.clear()
        ax_pred.clear()
        
        # 设置坐标轴
        for ax, title in [(ax_true, 'Ground Truth'), (ax_pred, 'Prediction')]:
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            ax.set_zlim([0, 10])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z (Height)')
            ax.set_title(f"{title} - Step: {frame}")
        
        # 绘制真实轨迹
        current_true = true_traj[frame]
        for i in range(num_objs):
            pos = current_true[i, 0:3]
            quat = current_true[i, 3:7]
            size = current_true[i, 7:10]
            
            if i == 0:  # 地面
                color = 'gray'
                alpha = 0.3
            else:
                color = colors[i % 10] if i in top_5_indices else "skyblue"
                alpha = 0.3
                
            vertices = get_cube_vertices(pos, size, quat)
            faces = get_cube_faces(vertices)
            poly = art3d.Poly3DCollection(faces, facecolors=color, linewidths=0.2, 
                                         edgecolors='black', alpha=alpha)
            poly.set_zorder(1)
            ax_true.add_collection3d(poly)
        
        # 绘制轨迹线 (真实)
        for i in top_5_indices:
            history = true_traj[:frame+1, i, 0:3]
            ax_true.plot(history[:, 0], history[:, 1], history[:, 2], 
                        color=colors[i % 10], linewidth=3, alpha=1.0, zorder=10)
        
        # 绘制质心 (真实)
        for i in range(num_objs):
            if current_true[i, 10] == 1:
                pos = current_true[i, 0:3]
                ax_true.scatter3D(pos[0], pos[1], pos[2], 
                               color='black', s=5, edgecolors='black', 
                               linewidth=0.8, depthshade=False, alpha=1.0, zorder=100)
        
        # 绘制预测轨迹
        current_pred = pred_traj[frame]
        for i in range(num_objs):
            pos = current_pred[i, 0:3]
            quat = current_pred[i, 3:7]
            size = current_pred[i, 7:10]
            
            if i == 0:  # 地面
                color = 'gray'
                alpha = 0.3
            else:
                color = colors[i % 10] if i in top_5_indices else "skyblue"
                alpha = 0.3
                
            vertices = get_cube_vertices(pos, size, quat)
            faces = get_cube_faces(vertices)
            poly = art3d.Poly3DCollection(faces, facecolors=color, linewidths=0.2, 
                                         edgecolors='black', alpha=alpha)
            poly.set_zorder(1)
            ax_pred.add_collection3d(poly)
        
        # 绘制轨迹线 (预测)
        for i in top_5_indices:
            history = pred_traj[:frame+1, i, 0:3]
            ax_pred.plot(history[:, 0], history[:, 1], history[:, 2], 
                        color=colors[i % 10], linewidth=3, alpha=1.0, zorder=10)
        
        # 绘制质心 (预测)
        for i in range(num_objs):
            if current_pred[i, 10] == 1:
                pos = current_pred[i, 0:3]
                ax_pred.scatter3D(pos[0], pos[1], pos[2], 
                               color='black', s=5, edgecolors='black', 
                               linewidth=0.8, depthshade=False, alpha=1.0, zorder=100)
        
        return ax_true, ax_pred

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)
    anim.save(save_path, writer='pillow')
    plt.close()
    print(f"Comparison animation saved to {save_path}")

def visualize_single_trajectory(traj_data, save_path, title="Trajectory", fps=25):
    """
    可视化单个轨迹 (用于只看真实或只看预测)
    
    Args:
        traj_data: [time, obj_num, 17]
        save_path: 保存路径
        title: 标题
        fps: 帧率
    """
    num_frames = traj_data.shape[0]
    num_objs = traj_data.shape[1]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, num_objs))

    # 识别需要显示轨迹线的物体
    initial_frame = traj_data[0]
    block_indices = []
    for i in range(num_objs):
        if initial_frame[i, 10] == 1:
            block_indices.append((i, initial_frame[i, 2]))
    
    block_indices.sort(key=lambda x: x[1], reverse=True)
    top_5_indices = [idx for idx, h in block_indices[:5]]
    
    def update(frame):
        ax.clear()
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([0, 10])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Height)')
        ax.set_title(f"{title} - Step: {frame}")
        
        current_frame = traj_data[frame]
        
        # 绘制积木
        for i in range(num_objs):
            pos = current_frame[i, 0:3]
            quat = current_frame[i, 3:7]
            size = current_frame[i, 7:10]
            
            if i == 0:
                color = 'gray'
                alpha = 0.3
            else:
                color = colors[i % 10] if i in top_5_indices else "skyblue"
                alpha = 0.3
                
            vertices = get_cube_vertices(pos, size, quat)
            faces = get_cube_faces(vertices)
            poly = art3d.Poly3DCollection(faces, facecolors=color, linewidths=0.2, 
                                         edgecolors='black', alpha=alpha)
            poly.set_zorder(1)
            ax.add_collection3d(poly)
        
        # 绘制轨迹线
        for i in top_5_indices:
            history = traj_data[:frame+1, i, 0:3]
            ax.plot(history[:, 0], history[:, 1], history[:, 2], 
                   color=colors[i % 10], linewidth=3, alpha=1.0, zorder=10)
        
        # 绘制质心
        for i in range(num_objs):
            if current_frame[i, 10] == 1:
                pos = current_frame[i, 0:3]
                ax.scatter3D(pos[0], pos[1], pos[2], 
                           color='black', s=5, edgecolors='black', 
                           linewidth=0.8, depthshade=False, alpha=1.0, zorder=100)
        
        return ax,

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)
    anim.save(save_path, writer='pillow')
    plt.close()
    print(f"Animation saved to {save_path}")

def visualize_from_npz(results_path, output_dir, fps=25):
    """
    从训练脚本保存的npz文件生成可视化
    
    Args:
        results_path: validation_*.npz 路径
        output_dir: 输出目录
        fps: 动画帧率
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载数据
    data = np.load(results_path, allow_pickle=True)
    stable_scenes = data['stable_scenes']
    unstable_scenes = data['unstable_scenes']
    
    print(f"Loaded validation data from: {results_path}")
    print(f"  Stable scenes: {len(stable_scenes)}")
    print(f"  Unstable scenes: {len(unstable_scenes)}")
    print(f"\nGenerating visualizations...")
    
    # 可视化stable场景
    for idx, scene in enumerate(stable_scenes):
        scene_name = scene['name'].replace('.npy', '')
        pred_traj = scene['pred']  # [time, obj, 17]
        true_traj = scene['true']  # [time, obj, 17]
        
        print(f"  Processing stable scene {idx+1}/{len(stable_scenes)}: {scene_name}")
        
        # 生成对比动画
        comparison_path = os.path.join(output_dir, f'stable_{idx}_{scene_name}_comparison.gif')
        visualize_comparison(pred_traj, true_traj, comparison_path, fps=fps)
        
        # 可选: 生成单独的预测和真实动画
        pred_path = os.path.join(output_dir, f'stable_{idx}_{scene_name}_pred.gif')
        true_path = os.path.join(output_dir, f'stable_{idx}_{scene_name}_true.gif')
        visualize_single_trajectory(pred_traj, pred_path, title="Prediction", fps=fps)
        visualize_single_trajectory(true_traj, true_path, title="Ground Truth", fps=fps)
    
    # 可视化unstable场景
    for idx, scene in enumerate(unstable_scenes):
        scene_name = scene['name'].replace('.npy', '')
        pred_traj = scene['pred']
        true_traj = scene['true']
        
        print(f"  Processing unstable scene {idx+1}/{len(unstable_scenes)}: {scene_name}")
        
        comparison_path = os.path.join(output_dir, f'unstable_{idx}_{scene_name}_comparison.gif')
        visualize_comparison(pred_traj, true_traj, comparison_path, fps=fps)
        
        pred_path = os.path.join(output_dir, f'unstable_{idx}_{scene_name}_pred.gif')
        true_path = os.path.join(output_dir, f'unstable_{idx}_{scene_name}_true.gif')
        visualize_single_trajectory(pred_traj, pred_path, title="Prediction", fps=fps)
        visualize_single_trajectory(true_traj, true_path, title="Ground Truth", fps=fps)
    
    print(f"\nVisualization complete! Results saved to: {output_dir}")
    print(f"Generated {3 * (len(stable_scenes) + len(unstable_scenes))} animations")
    print(f"  - Comparison animations: *_comparison.gif")
    print(f"  - Prediction only: *_pred.gif")
    print(f"  - Ground truth only: *_true.gif")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=False, default='exps/my_exp/validation_data/validation_best_epoch20.npz',
                       help='Path to validation_*.npz')
    parser.add_argument('--output_dir', type=str, default='exps/my_exp/animations',
                       help='Output directory for animations')
    parser.add_argument('--fps', type=int, default=25,
                       help='Animation frame rate')
    args = parser.parse_args()
    
    visualize_from_npz(
        results_path=args.results_path,
        output_dir=args.output_dir,
        fps=args.fps
    )

if __name__ == "__main__":
    main()
