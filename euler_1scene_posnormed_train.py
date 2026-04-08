import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import time
import numpy as np  
import logging
import argparse
import importlib

torch.autograd.set_detect_anomaly(True)

from utils.euler_blocktower_data_nff import DebugData, process_stacking_data_dynamic
from utils.util import vis_losscurve

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/nfs_project_a/chang/data_euler/data_euler/blocktower', help='Path to the dataset folder containing .npy files')
    parser.add_argument('--save_dir', type=str, default='exps/posnormed')
    parser.add_argument('--model_name', type=str, default='euler_neural_simulator')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta_min', type=float, default=1e-5, help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--layer_num', type=int, default=4)
    parser.add_argument('--segment_len', type=int, default=15, help='Number of simulation steps per segment, suggested 3-30 for training')
    parser.add_argument('--step_size', type=float, default=1/400, help='step size of ode solver (smaller = more accurate integration)')
    parser.add_argument('--dist_boundary', type=float, default=0.02, help='Boundary of distance mask')
    parser.add_argument('--use_dist_mask', action='store_true', default=True)
    parser.add_argument('--use_dist_input', action='store_true', default=True)
    parser.add_argument('--use_adjoint', action='store_true', default=False, help='Use adjoint method for memory efficiency')
    parser.add_argument('--euler_loss_weight', type=float, default=0.1, help='Weight for euler rotation loss relative to position loss')
    parser.add_argument('--scene_type', type=str, default='all', choices=['all', 'stable', 'unstable'], 
                       help='Filter dataset by scene type')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--val_interval', type=int, default=5, help='Run validation every N epochs')
    parser.add_argument('--save_vis_data', action='store_true', default=True, 
                       help='Save visualization data during validation')
    parser.add_argument('--vis_stable_scenes', type=int, default=3, 
                       help='Number of stable scenes to save for visualization')
    parser.add_argument('--vis_unstable_scenes', type=int, default=3, 
                       help='Number of unstable scenes to save for visualization')
    args = parser.parse_args()
    args.batch_size = 1  # 强制使用 batch_size=1，1-scene overfit
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate_epoch(model, val_loader, criterion, device, args, save_predictions=False):
    """
    在验证集上运行一个epoch
    
    Args:
        save_predictions: 是否保存预测结果用于可视化
    
    Returns:
        val_loss, val_loss_pos, val_loss_euler, predictions (如果save_predictions=True)
    """
    model.eval()
    val_loss = 0
    val_loss_pos = 0
    val_loss_euler = 0
    
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
            body_prop[..., 6:9] /= scale_view  # Size
            vel /= scale_view                   # Velocity

            # True Traj 用于算 Loss，也需要是归一化的
            true_traj = body_prop[..., 0:6].clone() # pos + euler
            
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
            pred_euler = pred_traj[..., 3:6]
            
            true_pos = true_traj[..., 0:3]
            true_euler = true_traj[..., 3:6]

            loss_pos = criterion(pred_pos, true_pos)
            loss_euler = criterion(pred_euler, true_euler)
            loss = loss_pos + args.euler_loss_weight * loss_euler

            val_loss += loss.item()
            val_loss_pos += loss_pos.item()
            val_loss_euler += loss_euler.item()
            
            # 保存可视化数据 - 分别收集stable和unstable场景
            if save_predictions and not collected_enough:
                # 这里的game_names是一个batch的场景名称列表
                batch_size = len(game_names)
                pred_traj_np = pred_traj.cpu().numpy()  # [batch, time, obj, 16]
                scale_np = scene_scale.cpu().numpy()  # [batch]
                true_traj_np = true_traj_orig_full.cpu().numpy() # 原始未归一化的真值 [Batch, Time, Obj, 16]

                # 反归一化用于可视化
                pred_traj_np[..., 0:3] *= scale_np[:, None, None, None]   # Position
                pred_traj_np[..., 6:9] *= scale_np[:, None, None, None] # Size
                pred_traj_np[..., 10:13] *= scale_np[:, None, None, None]  # Velocity
                
                for i in range(batch_size):
                    scene_name = game_names[i]
                    
                    # 判断场景类型
                    is_stable = 'stable' in scene_name and 'unstable' not in scene_name
                    is_unstable = 'unstable' in scene_name
                    
                    scene_data = {
                        'name': scene_name,
                        'pred': pred_traj_np[i],      # [time, obj, 16]
                        'true': true_traj_np[i],    # [time, obj, 16]
                        'num_objs': body_nums[i] if isinstance(body_nums, (list, np.ndarray)) else body_nums
                    }
                    
                    # 分类保存
                    if is_stable:
                        # 放宽限制：Debug时一律最多各存3个，不强制配平数量
                        if len(stable_scenes) < 3: stable_scenes.append(scene_data)
                    else:
                        if len(unstable_scenes) < 3: unstable_scenes.append(scene_data)
                    
                    # Debug模式下：只要存了我们想要的极简数据集(最多凑够6个用来看位置是否正确)，就够了
                    if len(stable_scenes) + len(unstable_scenes) >= 6:
                        if not collected_enough:
                            print(f"  Collected debug scenes for visualization: "
                                f"{len(stable_scenes)} stable, {len(unstable_scenes)} unstable")
                        collected_enough = True

    # 计算平均损失
    num_batches = len(val_loader)
    val_loss /= num_batches
    val_loss_pos /= num_batches
    val_loss_euler /= num_batches
    
    model.train()
    
    if save_predictions:
        # 合并场景数据
        predictions = {
            'stable_scenes': stable_scenes,
            'unstable_scenes': unstable_scenes
        }
        return val_loss, val_loss_pos, val_loss_euler, predictions
    else:
        return val_loss, val_loss_pos, val_loss_euler

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

    ForceFieldPredictor = model_module.ForceFieldPredictor
    ODEFunc = model_module.ODEFunc
    NeuralODEModel = model_module.NeuralODEModel
    
    force_predictor = ForceFieldPredictor(
        hidden_dim=args.hidden_dim, 
        output_layer=args.layer_num, 
        use_dist_mask=args.use_dist_mask,
        use_dist_input=args.use_dist_input,
        dist_boundary=args.dist_boundary
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
    dataset = DebugData(data_path=args.data_path, max_len=150, single_scene=True, block_cnt=2, scene_type='unstable')
    
    # 为了过拟合测试，训练集和验证集使用同一个绝对单一的数据集 (不划分)
    val_stable_indices, val_unstable_indices = [], [] # 占位符防报错
    
    # 创建训练集和验证集DataLoader (关闭shuffle，batch固定为1)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=args.eta_min)
    criterion = nn.MSELoss()
    
    print("Start Training...")
    if args.save_vis_data:
        print(f"Visualization data will be saved to: {vis_dir}")
        print(f"Will collect {args.vis_stable_scenes} stable and {args.vis_unstable_scenes} unstable scenes")
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    
    # 记录训练历史
    train_history = {'train_loss': [], 'val_loss': [], 'val_loss_pos': [], 'val_loss_euler': []}
    
    # Curriculum learning: 渐进增加segment长度，减少rollout误差累积
    curriculum_schedule = [
        (50, 5),      # epoch 0-49:  先用极短片段
        (150, 10),    # epoch 50-149: 逐渐增加到10帧
        (300, 15),    # epoch 150-299: 标准片段长度
        (args.epochs, 30),  # epoch 300+: 更长rollout
    ]

    for epoch in range(args.epochs):
        epoch_loss = 0
        start_time = time.time()

        current_segment_len = args.segment_len
        for seg_epoch, seg_len in curriculum_schedule:
            if epoch < seg_epoch:
                current_segment_len = seg_len
                break
        print(f"Epoch {epoch+1}: Training with Segment Length = {current_segment_len}")
        
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
            body_prop[..., 6:9] /= scale_view
            vel /= scale_view

            true_traj = body_prop[..., 0:6].clone()
            
            body_prop_s, vel_s, ang_vel_s, true_traj_s = process_stacking_data_dynamic(
                body_prop, true_traj, vel, ang_vel, SEGMENTS=current_segment_len
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
            pred_euler = pred_traj[..., 3:6]
            
            true_pos = true_traj_s[..., 0:3]
            true_euler = true_traj_s[..., 3:6]

            # 时间加权：后期帧权重更高（线性从0.5到1.5）
            n_steps = pred_pos.shape[1]
            time_w = torch.linspace(0.5, 1.5, n_steps, device=device).view(1, -1, 1, 1)
            loss_pos = torch.mean((pred_pos - true_pos) ** 2 * time_w)
            loss_euler = criterion(pred_euler, true_euler)
            loss = loss_pos + args.euler_loss_weight * loss_euler

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if batch_idx == 0 and (epoch + 1) % 5 == 0:
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
            #                  f"(Pos: {loss_pos:.6f}, Euler: {loss_euler:.6f}) ")
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {loss.item():.6f} "
                      f"(Pos: {loss_pos.item():.6f}, Euler: {loss_euler.item():.6f})")
        
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
            val_loss, val_loss_pos, val_loss_euler = validate_epoch(
                model, val_loader, criterion, device, args, save_predictions=False
            )
            
            train_history['val_loss'].append(val_loss)
            train_history['val_loss_pos'].append(val_loss_pos)
            train_history['val_loss_euler'].append(val_loss_euler)
            
            print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} (Pos: {val_loss_pos:.6f}, Euler: {val_loss_euler:.6f}) | "
                  f"Time: {time_elapsed:.2f}s")
            
            logging.info(
                f"[VAL] Epoch [{epoch+1}/{args.epochs}] Loss: {val_loss:.8f}, "
                f"MSE: {val_loss_pos:.8f}, Residual Loss: {val_loss_euler:.8f}, "
                f"Residual Residual Loss: 0.0"
            )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_best.pt'))
                print(f"  --> New best model saved! (Val Loss: {val_loss:.6f})")
                
            # [修改] 不管是不是best，只要需要保存数据，每 20 个 epochs 强制存一次，用于动态观察动作
            if args.save_vis_data and ((epoch + 1) % 10 == 0 or epoch == args.epochs - 1):
                print(f"  --> Collecting visualization data...")
                # 重新运行验证以收集可视化数据
                _, _, _, predictions = validate_epoch(
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
        
        scheduler.step()
        
        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pt'))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_final.pt'))
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'train_history.npz')
    np.savez(history_path, **train_history)
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Training history saved to: {history_path}")
    
    # 训练结束后，使用你 utils 里的函数自动画 Loss Curve (保存为图片)
    log_file_path = os.path.join(args.save_dir, 'train.log')
    try:
        vis_losscurve(steps=args.epochs, log_file=log_file_path)
    except Exception as e:
        print(f"Error drawing loss curve: {e}")

if __name__ == "__main__":
    main()
