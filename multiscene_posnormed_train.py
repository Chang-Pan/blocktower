import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np  
import sys
import logging
import argparse
import importlib

from utils.blocktower_data_nff import BlockTowerData, GroupedBatchSampler, process_stacking_data_dynamic
from utils.util import vis_losscurve, vis_lrcurve

def _normalized_quaternion_dot(pred_q, true_q):
    pred_q_n = F.normalize(pred_q, p=2, dim=-1, eps=1e-8)
    true_q_n = F.normalize(true_q, p=2, dim=-1, eps=1e-8)
    return (pred_q_n * true_q_n).sum(dim=-1, keepdim=True)


def quaternion_loss(pred_q, true_q, loss_type='mse', arccos_eps=1e-7):
    """Quaternion loss with selectable type: mse | stable | arccos | huber_angle | arccos_l1."""
    dot = _normalized_quaternion_dot(pred_q, true_q)

    if loss_type == 'mse':
        pred_q_aligned = torch.where(dot < 0, -pred_q, pred_q)
        return torch.mean((pred_q_aligned - true_q) ** 2)

    dot_abs = torch.abs(dot)
    if loss_type == 'stable':
        return torch.mean(1.0 - dot_abs)

    if loss_type == 'arccos':
        dot_abs = torch.clamp(dot_abs, min=0.0, max=1.0 - arccos_eps)
        angle_rad = 2.0 * torch.acos(dot_abs)
        return torch.mean(angle_rad ** 2)

    if loss_type == 'huber_angle':
        dot_abs = torch.clamp(dot_abs, min=0.0, max=1.0 - arccos_eps)
        angle = 2.0 * torch.acos(dot_abs)
        delta = 0.2  # ~11.5 degrees: below this use L2, above use L1
        loss = torch.where(
            angle < delta,
            0.5 * angle ** 2,
            delta * (angle - 0.5 * delta)
        )
        return torch.mean(loss)

    if loss_type == 'arccos_l1':
        dot_abs = torch.clamp(dot_abs, min=0.0, max=1.0 - arccos_eps)
        angle = 2.0 * torch.acos(dot_abs)
        return torch.mean(angle)

    raise ValueError(f"Unsupported quat_loss_type: {loss_type}")


def quaternion_angle_error_deg_stats(pred_q, true_q):
    """Return quaternion geodesic angle error stats in degrees (mean, p90, max)."""
    dot = _normalized_quaternion_dot(pred_q, true_q)
    dot_abs = torch.clamp(torch.abs(dot), min=0.0, max=1.0)
    angle_deg = 2.0 * torch.acos(dot_abs) * (180.0 / torch.pi)
    angle_deg_flat = angle_deg.reshape(-1)

    mean_deg = torch.mean(angle_deg_flat)
    p90_deg = torch.quantile(angle_deg_flat, 0.9)
    max_deg = torch.max(angle_deg_flat)
    return mean_deg, p90_deg, max_deg


def parse_curriculum_schedule(curriculum_arg, epochs, default_segment_len):
    """Parse preset or custom curriculum schedule into [(epoch_end, segment_len), ...]."""
    preset_schedules = {
        'default': [(50, 5), (150, 10), (300, 15), (epochs, 30)],
        'aggressive': [(30, 5), (100, 10), (200, 15), (epochs, 30)],
        'conservative': [(80, 5), (220, 10), (350, 15), (epochs, 30)],
        'none': [(epochs, default_segment_len)],
    }

    if curriculum_arg in preset_schedules:
        return preset_schedules[curriculum_arg]

    schedule = []
    for item in curriculum_arg.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(
                f"Invalid curriculum item '{item}'. Use 'epoch:segment' format."
            )
        epoch_str, seg_str = item.split(':', 1)
        epoch_end = int(epoch_str.strip())
        seg_len = int(seg_str.strip())
        if epoch_end <= 0 or seg_len <= 0:
            raise ValueError("Curriculum epoch and segment length must be positive.")
        schedule.append((epoch_end, seg_len))

    if not schedule:
        raise ValueError(
            "Curriculum is empty. Use preset (default/aggressive/conservative/none) "
            "or custom format like '50:5,150:10,300:15,500:30'."
        )

    schedule.sort(key=lambda x: x[0])
    if schedule[-1][0] < epochs:
        schedule.append((epochs, schedule[-1][1]))
    return schedule


def build_optimizer(args, model):
    if args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    # 构建主调度器
    if args.scheduler == 'coswr':
        main_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=args.eta_min
        )
    elif args.scheduler == 'cosine':
        # 若有 warmup, cosine 的 T_max 应为剩余 epoch
        t_max = args.epochs - args.warmup_epochs if args.warmup_epochs > 0 else args.epochs
        main_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=args.eta_min
        )
    elif args.scheduler == 'step':
        main_sched = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif args.scheduler == 'none':
        main_sched = None
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    # warmup: 前 warmup_epochs 个 epoch 线性从 lr*warmup_start_factor 爬到 lr
    if args.warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        if main_sched is None:
            return warmup_sched
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[args.warmup_epochs],
        )

    return main_sched

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/nfs_project_a/chang/small_data/data/blocktower', help='Path to the dataset folder containing .npy files')
    parser.add_argument('--save_dir', type=str, default='exps/posnormed')
    parser.add_argument('--model_name', type=str, default='posnormed_neural_simulator')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta_min', type=float, default=1e-5, help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--layer_num', type=int, default=4)
    # SlotFormer-specific args (ignored by neural_simulator)
    parser.add_argument('--history_len', type=int, default=1, help='[SlotFormer] burn-in frames')
    parser.add_argument('--num_heads', type=int, default=8, help='[SlotFormer] transformer attention heads')
    parser.add_argument('--ffn_dim', type=int, default=512, help='[SlotFormer] transformer feedforward dim')
    parser.add_argument('--slotres_scale', type=float, default=1e2, help='[SlotFormer] residual scale; larger = smaller per-step state change')
    parser.add_argument('--segment_len', type=int, default=15, help='Number of simulation steps per segment, suggested 3-30 for training')
    parser.add_argument('--segment_stride', type=int, default=0, help='Stride for segment slicing; <=0 means use segment_len (no overlap)')
    parser.add_argument('--step_size', type=float, default=1/400, help='step size of ode solver (smaller = more accurate integration)')
    parser.add_argument('--dist_boundary', type=float, default=0.02, help='Boundary of distance mask')
    parser.add_argument('--use_dist_mask', action='store_true', default=True)
    parser.add_argument('--use_dist_input', action='store_true', default=True)
    parser.add_argument('--use_adjoint', action='store_true', default=True, help='Use adjoint method for memory efficiency')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='coswr', choices=['coswr', 'cosine', 'step', 'none'],
                       help='LR scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Linear warmup epochs before main scheduler (0 = no warmup)')
    parser.add_argument('--warmup_start_factor', type=float, default=0.01,
                       help='Warmup starts at lr * this factor, linearly ramps to lr')
    parser.add_argument('--curriculum', type=str, default='default',
                       help='Curriculum preset(default/aggressive/conservative/none) or custom "50:5,150:10,..."')
    parser.add_argument(
        '--quat_loss_type',
        type=str,
        default='mse',
        choices=['mse', 'stable', 'arccos', 'huber_angle', 'arccos_l1'],
        help='Quaternion loss type: mse(sign-aware) | stable(1-|dot|) | arccos((2*acos(|dot|))^2) | huber_angle(Huber on angle, delta=0.2) | arccos_l1(mean angle)'
    )
    parser.add_argument(
        '--quat_loss_weight',
        type=float,
        default=0.1,
        help='Weight for quaternion/rotation loss. Recommended candidates: mse={0.1,0.3,1.0}, stable={0.3,1.0,3.0}, arccos={0.01,0.03,0.1}'
    )
    parser.add_argument(
        '--arccos_eps',
        type=float,
        default=1e-7,
        help='Epsilon for acos clamp: dot_abs clamped to [0, 1-eps]. Larger eps = smaller max gradient. 1e-7(default) -> grad~7071, 1e-4 -> grad~100, 1e-2 -> grad~10'
    )
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--force_tanh', action='store_true', default=False,
                       help='Bound force output with tanh * learnable scale (prevents ODE divergence)')
    parser.add_argument('--scene_type', type=str, default='all', choices=['all', 'stable', 'unstable'], 
                       help='Filter dataset by scene type')
    parser.add_argument('--min_blocks', type=int, default=None, help='Min block count (inclusive); None = no limit')
    parser.add_argument('--max_blocks', type=int, default=None, help='Max block count (inclusive); None = no limit. e.g. --max_blocks 5 for 1-5 block subset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--val_interval', type=int, default=10, help='Run validation every N epochs')
    parser.add_argument('--save_vis_data', action='store_true', default=True, 
                       help='Save visualization data during validation')
    parser.add_argument('--vis_stable_scenes', type=int, default=3, 
                       help='Number of stable scenes to save for visualization')
    parser.add_argument('--vis_unstable_scenes', type=int, default=3, 
                       help='Number of unstable scenes to save for visualization')
    args = parser.parse_args()
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate_epoch(model, val_loader, criterion, device, args, save_predictions=False):
    """
    在验证集上运行一个epoch
    
    Args:
        save_predictions: 是否保存预测结果用于可视化
    
    Returns:
        val_loss, val_loss_pos, val_loss_quat, val_angle_mean, val_angle_p90, val_angle_max,
        predictions (如果save_predictions=True)
    """
    model.eval()
    val_loss = 0
    val_loss_pos = 0
    val_loss_quat = 0
    angle_error_all = []
    
    # 如果需要保存可视化数据,分别收集stable和unstable场景
    stable_scenes = [] if save_predictions else None
    unstable_scenes = [] if save_predictions else None
    
    collected_enough = False

    with torch.no_grad():
        for batch_idx, (game_names, body_prop, vel, ang_vel, body_nums) in enumerate(val_loader):
            body_prop = body_prop.to(device)
            vel = vel.to(device)
            ang_vel = ang_vel.to(device)
            
            # 全局归一化
            # 计算scale：找每个batch里最大的坐标值，只考虑第0帧
            pos_initial = body_prop[:, 0, :, 0:3] # [Batch, Obj, 3]
            pos_flat = pos_initial.reshape(body_prop.size(0), -1) # [Batch, Time*Obj*3]
            scene_scale = torch.max(torch.abs(pos_flat), dim=1)[0] # [Batch]
            scene_scale = torch.clamp(scene_scale, min=1.0) # 防止过小
            scale_view = scene_scale.view(-1, 1, 1, 1) # [batch, 1, 1, 1]

            # 备份原始数据用于 True Trajectory (在保存可视化时用到)
            # 注意: 深拷贝一份以防被原地修改
            true_traj_orig_full = torch.cat([body_prop, vel, ang_vel], dim=-1).clone() # [Batch, Time, Obj, 17]

            # 执行归一化 (克隆避免原地修改污染原数据集缓存)
            body_prop = body_prop.clone()
            vel = vel.clone()
            
            body_prop[..., 0:3] /= scale_view   # Position
            body_prop[..., 7:10] /= scale_view  # Size
            vel /= scale_view                   # Velocity

            # True Traj 用于算 Loss，也需要是归一化的
            true_traj = body_prop[..., 0:7].clone() # pos + quat
            
            # 第0帧作为初始状态
            z0 = torch.cat([
                body_prop[:, 0, :, :], 
                vel[:, 0, :, :],       
                ang_vel[:, 0, :, :]    
            ], dim=-1)
            
            sim_steps = true_traj.shape[1] # 150
            t = torch.linspace(0, (sim_steps-1)/25.0, steps=sim_steps, device=device).unsqueeze(0)
            
            # pred_traj的输出是归一化的，所以在计算loss时直接用pred_traj和true_traj（都是归一化的）进行比较是合理的
            pred_traj = model(z0, t, scene_scale=scene_scale)  # [batch, time, obj, 17]
            pred_pos = pred_traj[..., 0:3]
            pred_quat = pred_traj[..., 3:7]
            
            true_pos = true_traj[..., 0:3]
            true_quat = true_traj[..., 3:7]

            loss_pos = criterion(pred_pos, true_pos)
            loss_quat = quaternion_loss(
                pred_quat,
                true_quat,
                loss_type=args.quat_loss_type,
                arccos_eps=args.arccos_eps,
            )
            loss = loss_pos + args.quat_loss_weight * loss_quat

            batch_angle_deg = 2.0 * torch.acos(
                torch.clamp(torch.abs(_normalized_quaternion_dot(pred_quat, true_quat)), min=0.0, max=1.0)
            ) * (180.0 / torch.pi)
            angle_error_all.append(batch_angle_deg.reshape(-1).cpu())

            val_loss += loss.item()
            val_loss_pos += loss_pos.item()
            val_loss_quat += loss_quat.item()
            
            # 保存可视化数据 - 分别收集stable和unstable场景
            if save_predictions and not collected_enough:
                # 这里的game_names是一个batch的场景名称列表
                batch_size = len(game_names)
                pred_traj_np = pred_traj.cpu().numpy()  # [batch, time, obj, 17]
                scale_np = scene_scale.cpu().numpy()  # [batch]
                true_traj_np = true_traj_orig_full.cpu().numpy() # 原始未归一化的真值 [Batch, Time, Obj, 17]

                # 反归一化用于可视化
                pred_traj_np[..., 0:3] *= scale_np[:, None, None, None]   # Position
                pred_traj_np[..., 7:10] *= scale_np[:, None, None, None] # Size
                pred_traj_np[..., 11:14] *= scale_np[:, None, None, None]  # Velocity
                
                for i in range(batch_size):
                    scene_name = game_names[i]
                    
                    # 判断场景类型
                    is_stable = 'stable' in scene_name and 'unstable' not in scene_name
                    is_unstable = 'unstable' in scene_name
                    
                    scene_data = {
                        'name': scene_name,
                        'pred': pred_traj_np[i],      # [time, obj, 17]
                        'true': true_traj_np[i],    # [time, obj, 17]
                        'num_objs': body_nums[i] if isinstance(body_nums, (list, np.ndarray)) else body_nums
                    }
                    
                    # 分类保存
                    if is_stable:
                        if len(stable_scenes) < args.vis_stable_scenes: stable_scenes.append(scene_data)
                    else:
                        if len(unstable_scenes) < args.vis_unstable_scenes: unstable_scenes.append(scene_data)
                    
                    target_total = args.vis_stable_scenes + args.vis_unstable_scenes
                    if len(stable_scenes) + len(unstable_scenes) >= target_total:
                        if not collected_enough:
                            print(f"  Collected debug scenes for visualization: "
                                f"{len(stable_scenes)} stable, {len(unstable_scenes)} unstable")
                        collected_enough = True

    # 计算平均损失
    num_batches = len(val_loader)
    val_loss /= num_batches
    val_loss_pos /= num_batches
    val_loss_quat /= num_batches
    if len(angle_error_all) > 0:
        all_angle = torch.cat(angle_error_all)
        val_angle_mean = torch.mean(all_angle).item()
        val_angle_p90 = torch.quantile(all_angle, 0.9).item()
        val_angle_max = torch.max(all_angle).item()
    else:
        val_angle_mean = 0.0
        val_angle_p90 = 0.0
        val_angle_max = 0.0
    
    model.train()
    
    if save_predictions:
        # 合并场景数据
        predictions = {
            'stable_scenes': stable_scenes,
            'unstable_scenes': unstable_scenes
        }
        return val_loss, val_loss_pos, val_loss_quat, val_angle_mean, val_angle_p90, val_angle_max, predictions
    else:
        return val_loss, val_loss_pos, val_loss_quat, val_angle_mean, val_angle_p90, val_angle_max

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # 创建可视化数据保存目录
    vis_dir = os.path.join(args.save_dir, 'validation_data')
    if args.save_vis_data and not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        
    logging.basicConfig(filename=os.path.join(args.save_dir, 'train.log'), level=logging.INFO)

    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"Training started at {time_str}")
    logging.info(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. 加载模型
    try:
        model_module = importlib.import_module(args.model_name)
    except ImportError:
        # 兼容传入短名: euler_neural_simulator -> models.euler_neural_simulator
        try:
            model_module = importlib.import_module(f"models.{args.model_name}")
        except ImportError as e:
            raise ImportError(
                f"Cannot import model '{args.model_name}'. "
                f"Try --model_name models.euler_neural_simulator"
            ) from e

    print(f"Loaded model module: {model_module.__name__}")    

    module_short_name = model_module.__name__.split(".")[-1]
    if "slotformer" in module_short_name:
        # Transformer-based dynamics model (no ODE integration -> no divergence NaN)
        model = model_module.DynamicsSlotFormer(
            slot_size=17,
            history_len=args.history_len,
            d_model=args.hidden_dim,
            num_layers=args.layer_num,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            norm_first=True,
            slotres_scale=args.slotres_scale,
        )
        model.to(device)
    else:
        # Neural ODE force-field model
        ForceFieldPredictor = model_module.ForceFieldPredictor
        ODEFunc = model_module.ODEFunc
        NeuralODEModel = model_module.NeuralODEModel

        force_predictor = ForceFieldPredictor(
            hidden_dim=args.hidden_dim, 
            output_layer=args.layer_num, 
            use_dist_mask=args.use_dist_mask,
            use_dist_input=args.use_dist_input,
            dist_boundary=args.dist_boundary,
            force_tanh=args.force_tanh
        )
        ode_func = ODEFunc(force_predictor, mass=1.0)
        model = NeuralODEModel(ode_func, use_adjoint=args.use_adjoint, step_size=args.step_size)
        model.to(device)

    print("Model initialized successfully.")
    print(f"Total trainable parameters: {count_parameters(model)}")
    print("-" * 50)

    model.train()
    
    # 2. 数据加载和划分
    print(f"Loading Dataset ({args.scene_type})...")
    dataset = BlockTowerData(data_path=args.data_path, max_len=150, scene_type=args.scene_type,
                             min_blocks=args.min_blocks, max_blocks=args.max_blocks)

    # 划分训练集和验证集 (按 stable/unstable 分层)
    train_dataset, val_dataset, val_stable_indices, val_unstable_indices = dataset.split_train_val(
        val_ratio=args.val_ratio, seed=args.seed
    )

    # 创建 GroupedBatchSampler (按积木数分组，同 batch 内积木数相同)
    train_sampler = GroupedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_sampler = GroupedBatchSampler(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_workers=0
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=num_workers)

    # 在 train_loader 创建之后、训练循环之前加这段
    print("=== DataLoader sanity check ===")
    print(f"Configured batch_size = {args.batch_size}")
    for i, (game_names, body_prop, vel, ang_vel, body_nums) in enumerate(train_loader):
        print(
            f"Batch {i}: actual_batch_size={len(game_names)}, "
            f"body_prop={body_prop.shape}, vel={vel.shape}, ang_vel={ang_vel.shape}, "
            f"body_nums={body_nums}"
        )
        if i >= 5:
            break
    print("===============================")

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.MSELoss()
    
    print("Start Training...")
    if args.save_vis_data:
        print(f"Visualization data will be saved to: {vis_dir}")
        print(f"Will collect {args.vis_stable_scenes} stable and {args.vis_unstable_scenes} unstable scenes")
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    best_epoch = -1
    best_val_pos = float('inf')
    best_val_quat = float('inf')
    best_val_angle_mean = float('inf')
    
    # 记录训练历史
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'val_loss_pos': [],
        'val_loss_quat': [],
        'val_angle_mean_deg': [],
        'val_angle_p90_deg': [],
        'val_angle_max_deg': [],
    }
    
    # Curriculum learning: 渐进增加segment长度，减少rollout误差累积
    # (epoch_end, segment_len): 在epoch < epoch_end时使用该segment_len
    curriculum_schedule = parse_curriculum_schedule(
        curriculum_arg=args.curriculum,
        epochs=args.epochs,
        default_segment_len=args.segment_len,
    )

    for epoch in range(args.epochs):
        epoch_loss = 0
        start_time = time.time()

        current_segment_len = args.segment_len
        current_segment_stride = args.segment_len if args.segment_stride <= 0 else args.segment_stride
        for seg_epoch, seg_len in curriculum_schedule:
            if epoch < seg_epoch:
                current_segment_len = seg_len
                if args.segment_stride <= 0:
                    current_segment_stride = current_segment_len
                break
        if args.segment_stride > 0:
            current_segment_stride = min(current_segment_stride, current_segment_len)
        print(f"Epoch {epoch+1}: Training with Segment Length = {current_segment_len}, Stride = {current_segment_stride}")
        
        # 训练阶段
        for batch_idx, (game_names, body_prop, vel, ang_vel, body_nums) in enumerate(train_loader):
            
            body_prop = body_prop.to(device)
            vel = vel.to(device)
            ang_vel = ang_vel.to(device)
            
            # [新增] 归一化输入数据
            # 计算当前 batch 的 global scale
            # 注意: 这里 body_prop 是 [Batch, Time, Obj, Feat]
            pos_initial = body_prop[:, 0, :, 0:3] # [Batch, Obj, 3]
            pos_flat = pos_initial.reshape(body_prop.size(0), -1)
            scene_scale = torch.max(torch.abs(pos_flat), dim=1)[0]
            scene_scale = torch.clamp(scene_scale, min=1.0)
            scale_view = scene_scale.view(-1, 1, 1, 1)

            body_prop = body_prop.clone()
            vel = vel.clone()
            
            body_prop[..., 0:3] /= scale_view
            body_prop[..., 7:10] /= scale_view
            vel /= scale_view

            true_traj = body_prop[..., 0:7].clone()
            
            body_prop_s, vel_s, ang_vel_s, true_traj_s = process_stacking_data_dynamic(
                body_prop, true_traj, vel, ang_vel, SEGMENTS=current_segment_len, STRIDE=current_segment_stride
            )

            # process_stacking_data_dynamic 会展平 batch 和 segments -> [Batch*Seg, Time, Obj, ...]
            # 我们的 scene_scale 是 [Batch], 需要扩展对应
            num_segments_per_sample = body_prop_s.shape[0] // body_prop.shape[0] # 每条数据切成了几段
            scene_scale_expanded = scene_scale.repeat_interleave(num_segments_per_sample) # [Batch*Seg]
            
            z0 = torch.cat([
                body_prop_s[:, 0, :, :], 
                vel_s[:, 0, :, :],       
                ang_vel_s[:, 0, :, :]    
            ], dim=-1)
            
            sim_steps = true_traj_s.shape[1]
            t = torch.linspace(0, (sim_steps-1)/25.0, steps=sim_steps, device=device).unsqueeze(0)
            
            optimizer.zero_grad()
            
            pred_traj = model(z0, t, scene_scale=scene_scale_expanded)
            pred_pos = pred_traj[..., 0:3]
            pred_quat = pred_traj[..., 3:7]
            
            true_pos = true_traj_s[..., 0:3]
            true_quat = true_traj_s[..., 3:7]

            # 时间加权：后期帧权重更高（线性从0.5到1.5）
            n_steps = pred_pos.shape[1]
            time_w = torch.linspace(0.5, 1.5, n_steps, device=device).view(1, -1, 1, 1)
            loss_pos = torch.mean((pred_pos - true_pos) ** 2 * time_w)
            loss_quat = quaternion_loss(
                pred_quat,
                true_quat,
                loss_type=args.quat_loss_type,
                arccos_eps=args.arccos_eps,
            )
            loss = loss_pos + args.quat_loss_weight * loss_quat

            # NaN early stopping: 一旦检测到NaN立刻终止训练
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n{'='*60}")
                print(f"[FATAL] NaN/Inf detected in loss at Epoch {epoch+1}, Batch {batch_idx+1}")
                print(f"  loss={loss.item()}, loss_pos={loss_pos.item()}, loss_quat={loss_quat.item()}")
                print(f"  pred_pos has NaN: {torch.isnan(pred_pos).any().item()}")
                print(f"  pred_quat has NaN: {torch.isnan(pred_quat).any().item()}")
                # 毒性 batch 诊断
                print(f"  --- Batch diagnostics ---")
                print(f"  body_nums: {body_nums}")
                print(f"  scene_names: {game_names}")                
                print(f"  pred_pos range: [{pred_pos.min().item():.4f}, {pred_pos.max().item():.4f}]")
                print(f"  pred_quat range: [{pred_quat.min().item():.4f}, {pred_quat.max().item():.4f}]")
                if not torch.isnan(pred_pos).all():
                    print(f"  pred_pos abs max: {pred_pos[~torch.isnan(pred_pos)].abs().max().item():.4f}")
                print(f"  Training terminated early.")
                print(f"  Best model so far: Epoch {best_epoch}, Val Loss {best_val_loss:.6f}")
                print(f"{'='*60}")
                                # 保存 NaN 之前的 history 记录和图片
                try:
                    # 获取日志路径
                    log_file_path = os.path.join(args.save_dir, 'train.log')
                    
                    # 只要已经完成过至少 1 个 epoch，就可以画图
                    if epoch > 0:
                        vis_losscurve(steps=epoch, log_file=log_file_path)
                        vis_lrcurve(log_file=log_file_path)
                    
                    # 顺便把已有的 train_history 保存下来，方便之后查看详细数值
                    history_path = os.path.join(args.save_dir, 'train_history_nan_earlystop.npz')
                    np.savez(history_path, **train_history)
                    print(f"  Saved partial history to {history_path}")
                except Exception as e:
                    print(f"  Failed to draw partial curves or save history: {e}")

                sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if batch_idx == 0 and (epoch + 1) % 5 == 0 and hasattr(model, 'ode_func'):
                fp = model.ode_func.force_predictor

                b_grad = fp.branch_net[0].weight.grad
                t_grad = fp.trunk_net[0].weight.grad

                # ground_mlp: [Linear, ReLU, Linear, ReLU, Linear]
                g0_grad = fp.ground_mlp[0].weight.grad   # first linear
                g2_grad = fp.ground_mlp[2].weight.grad   # middle linear
                g4_grad = fp.ground_mlp[4].weight.grad   # last linear

                b_norm = b_grad.norm().item() if b_grad is not None else 0.0
                t_norm = t_grad.norm().item() if t_grad is not None else 0.0
                g0_norm = g0_grad.norm().item() if g0_grad is not None else 0.0
                g2_norm = g2_grad.norm().item() if g2_grad is not None else 0.0
                g4_norm = g4_grad.norm().item() if g4_grad is not None else 0.0

                print(
                    f"[Grad] branch_l0={b_norm:.6e} | trunk_l0={t_norm:.6e} | "
                    f"ground_l0={g0_norm:.6e} | ground_l1={g2_norm:.6e} | ground_l2={g4_norm:.6e}"
                )
            
            optimizer.step()
            
            epoch_loss += loss.item()

            # logging.info(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f} | "
            #                  f"(Pos: {loss_pos:.6f}, Quat: {loss_quat:.6f}) ")
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {loss.item():.6f} "
                      f"(Pos: {loss_pos.item():.6f}, Quat: {loss_quat.item():.6f})")
        
        avg_train_loss = epoch_loss / len(train_loader)
        time_elapsed = time.time() - start_time
        train_history['train_loss'].append(avg_train_loss)

        # 新增: 每个 epoch 都记录 TRAIN loss（供画图分离）
        logging.info(
            f"[TRAIN] Epoch [{epoch+1}/{args.epochs}] Loss: {avg_train_loss:.8f}, "
            f"MSE: {avg_train_loss:.8f}, Residual Loss: 0.0, Residual Residual Loss: 0.0"
        )

        # 验证阶段
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            # 先运行验证计算损失
            val_loss, val_loss_pos, val_loss_quat, val_angle_mean, val_angle_p90, val_angle_max = validate_epoch(
                model, val_loader, criterion, device, args, save_predictions=False
            )
            
            train_history['val_loss'].append(val_loss)
            train_history['val_loss_pos'].append(val_loss_pos)
            train_history['val_loss_quat'].append(val_loss_quat)
            train_history['val_angle_mean_deg'].append(val_angle_mean)
            train_history['val_angle_p90_deg'].append(val_angle_p90)
            train_history['val_angle_max_deg'].append(val_angle_max)
            
            print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} (Pos: {val_loss_pos:.6f}, Quat: {val_loss_quat:.6f}) | "
                  f"AngleDeg(mean/p90/max): {val_angle_mean:.3f}/{val_angle_p90:.3f}/{val_angle_max:.3f} | "
                  f"Time: {time_elapsed:.2f}s")
            
            logging.info(
                f"[VAL] Epoch [{epoch+1}/{args.epochs}] Loss: {val_loss:.8f}, "
                f"MSE: {val_loss_pos:.8f}, Residual Loss: {val_loss_quat:.8f}, "
                f"Residual Residual Loss: 0.0, "
                f"Angle Mean Deg: {val_angle_mean:.6f}, Angle P90 Deg: {val_angle_p90:.6f}, Angle Max Deg: {val_angle_max:.6f}"
            )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_val_pos = val_loss_pos
                best_val_quat = val_loss_quat
                best_val_angle_mean = val_angle_mean
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_best.pt'))
                print(
                    f"  --> New best model saved! (Epoch: {best_epoch}, Val Loss: {val_loss:.6f}, "
                    f"Pos: {best_val_pos:.6f}, Quat: {best_val_quat:.6f}, AngleMeanDeg: {best_val_angle_mean:.3f})"
                )

                # 只在 val loss 创新低时收集可视化数据
                if args.save_vis_data:
                    print(f"  --> Collecting visualization data...")
                    _, _, _, _, _, _, predictions = validate_epoch(
                        model, val_loader, criterion, device, args, save_predictions=True
                    )
                    
                    vis_save_path = os.path.join(vis_dir, f'debug_vis_epoch{epoch+1}.npz')
                    np.savez(vis_save_path,
                            stable_scenes=predictions['stable_scenes'],
                            unstable_scenes=predictions['unstable_scenes'],
                            epoch=epoch+1,
                            val_loss=val_loss,
                            args=vars(args))
                    print(f"  --> Visualization data saved: {vis_save_path}")
                    print(f"      ({len(predictions['stable_scenes'])} stable, {len(predictions['unstable_scenes'])} unstable scenes)")
                
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.6f} | "
                  f"Time: {time_elapsed:.2f}s")

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"[LR] Epoch [{epoch+1}/{args.epochs}] LR: {current_lr:.10e}")
        
        if scheduler is not None:
            scheduler.step()
        
        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pt'))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_final.pt'))
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'train_history.npz')
    np.savez(history_path, **train_history)
    
    print(
        f"\nTraining completed! Best validation loss: {best_val_loss:.6f} "
        f"(Epoch: {best_epoch}, Pos: {best_val_pos:.6f}, Quat: {best_val_quat:.6f}, AngleMeanDeg: {best_val_angle_mean:.3f})"
    )
    print(f"Training history saved to: {history_path}")
    
    # 训练结束后，使用你 utils 里的函数自动画 Loss Curve (保存为图片)
    log_file_path = os.path.join(args.save_dir, 'train.log')
    try:
        vis_losscurve(steps=args.epochs, log_file=log_file_path)
    except Exception as e:
        print(f"Error drawing loss curve: {e}")

    try:
        vis_lrcurve(log_file=log_file_path)
    except Exception as e:
        print(f"Error drawing lr curve: {e}")

if __name__ == "__main__":
    main()
