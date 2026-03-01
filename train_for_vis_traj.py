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

from utils.blocktower_data_nff import BlockTowerData, GroupedBatchSampler, process_stacking_data_dynamic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/nfs_project_a/chang/data/data/blocktower', help='Path to the dataset folder containing .npy files')
    parser.add_argument('--save_dir', type=str, default='exps/my_exp')
    parser.add_argument('--model_name', type=str, default='neural_simulator')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--segment_len', type=int, default=15, help='Number of simulation steps per segment, suggested 3-30 for training')
    parser.add_argument('--step_size', type=float, default=1/200, help='step size of ode solver')
    parser.add_argument('--dist_boundary', type=float, default=0.03, help='Boundary of distance mask')
    parser.add_argument('--use_dist_mask', action='store_true', default=True)
    parser.add_argument('--use_dist_input', action='store_true', default=True)
    parser.add_argument('--use_adjoint', action='store_true', default=True, help='Use adjoint method for memory efficiency')
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
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate_epoch(model, val_loader, criterion, device, args, save_predictions=False):
    """
    在验证集上运行一个epoch
    
    Args:
        save_predictions: 是否保存预测结果用于可视化
    
    Returns:
        val_loss, val_loss_pos, val_loss_quat, predictions (如果save_predictions=True)
    """
    model.eval()
    val_loss = 0
    val_loss_pos = 0
    val_loss_quat = 0
    
    # 如果需要保存可视化数据,分别收集stable和unstable场景
    stable_scenes = [] if save_predictions else None
    unstable_scenes = [] if save_predictions else None
    
    with torch.no_grad():
        for batch_idx, (game_names, body_prop, vel, ang_vel, body_nums) in enumerate(val_loader):
            body_prop = body_prop.to(device)
            vel = vel.to(device)
            ang_vel = ang_vel.to(device)
            
            true_traj = body_prop[..., 0:7].clone() # pos + quat
            
            # 第0帧作为初始状态
            z0 = torch.cat([
                body_prop[:, 0, :, :], 
                vel[:, 0, :, :],       
                ang_vel[:, 0, :, :]    
            ], dim=-1)
            
            sim_steps = true_traj.shape[1] # 150
            t = torch.linspace(0, (sim_steps-1)/25.0, steps=sim_steps, device=device).unsqueeze(0)
            
            pred_traj = model(z0, t)  # [batch, time, obj, 17]
            pred_pos = pred_traj[..., 0:3]
            pred_quat = pred_traj[..., 3:7]
            
            true_pos = true_traj[..., 0:3]
            true_quat = true_traj[..., 3:7]

            loss_pos = criterion(pred_pos, true_pos)
            loss_quat = criterion(pred_quat, true_quat)
            loss = loss_pos + loss_quat
            
            val_loss += loss.item()
            val_loss_pos += loss_pos.item()
            val_loss_quat += loss_quat.item()
            
            # 保存可视化数据 - 分别收集stable和unstable场景
            if save_predictions:
                # 这里的game_names是一个batch的场景名称列表
                batch_size = len(game_names)
                pred_traj_np = pred_traj.cpu().numpy()  # [batch, time, obj, 17]
                
                # 构建完整的true_traj (包含速度和角速度)
                # true_traj 只有 [batch, time, obj, 7], 需要补充速度
                true_traj_full = torch.cat([
                    true_traj,      # [batch, time, obj, 7] pos + quat
                    body_prop[..., 7:11],  # [batch, time, obj, 4] size + dynamic_mask
                    vel,            # [batch, time, obj, 3] velocity
                    ang_vel         # [batch, time, obj, 3] angular velocity
                ], dim=-1).cpu().numpy()  # [batch, time, obj, 17]
                
                for i in range(batch_size):
                    scene_name = game_names[i]
                    
                    # 判断场景类型
                    is_stable = 'stable' in scene_name and 'unstable' not in scene_name
                    is_unstable = 'unstable' in scene_name
                    
                    scene_data = {
                        'name': scene_name,
                        'pred': pred_traj_np[i],      # [time, obj, 17]
                        'true': true_traj_full[i],    # [time, obj, 17]
                        'num_objs': body_nums[i] if isinstance(body_nums, (list, np.ndarray)) else body_nums
                    }
                    
                    # 分类保存
                    if is_stable and len(stable_scenes) < args.vis_stable_scenes:
                        stable_scenes.append(scene_data)
                    elif is_unstable and len(unstable_scenes) < args.vis_unstable_scenes:
                        unstable_scenes.append(scene_data)
                    
                    # 如果已经收集够了,可以提前结束
                    if (len(stable_scenes) >= args.vis_stable_scenes and 
                        len(unstable_scenes) >= args.vis_unstable_scenes):
                        break
            
            # 提前结束循环
            if save_predictions and (len(stable_scenes) >= args.vis_stable_scenes and 
                                     len(unstable_scenes) >= args.vis_unstable_scenes):
                print(f"  Collected enough scenes for visualization: "
                      f"{len(stable_scenes)} stable, {len(unstable_scenes)} unstable")
                break
    
    # 计算平均损失
    num_batches = len(val_loader)
    val_loss /= num_batches
    val_loss_pos /= num_batches
    val_loss_quat /= num_batches
    
    model.train()
    
    if save_predictions:
        # 合并场景数据
        predictions = {
            'stable_scenes': stable_scenes,
            'unstable_scenes': unstable_scenes
        }
        return val_loss, val_loss_pos, val_loss_quat, predictions
    else:
        return val_loss, val_loss_pos, val_loss_quat

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
        import models.neural_simulator as model_module
        
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
    dataset = BlockTowerData(data_path=args.data_path, max_len=150, scene_type=args.scene_type)
    
    # 划分训练集和验证集
    train_dataset, val_dataset, val_stable_indices, val_unstable_indices = dataset.split_train_val(
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # 创建训练集DataLoader
    train_batch_sampler = GroupedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=0)
    
    # 创建验证集DataLoader
    val_batch_sampler = GroupedBatchSampler(val_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=0)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    print("Start Training...")
    if args.save_vis_data:
        print(f"Visualization data will be saved to: {vis_dir}")
        print(f"Will collect {args.vis_stable_scenes} stable and {args.vis_unstable_scenes} unstable scenes")
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    
    # 记录训练历史
    train_history = {'train_loss': [], 'val_loss': [], 'val_loss_pos': [], 'val_loss_quat': []}
    
    curriculum_schedule = [
        (100, args.segment_len),   # 0-99, len=segment_len
    ]
    #curriculum_schedule = [
        #(50, 5),   # 0-49, len=5
        #(100, 10), # 50-99, len=10
        #(150, 15), # 100-149, len=15
        #(200, 20), # 150-199, len=20
        #(250, 5), # 200-249, len=5
        #(300, 10), # 250-299, len=10
        #(350, 15), # 300-349, len=15
        #(400, 20), # 350-399, len=20
        #(450, 30), # 400-449, len=30
    #]

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
            
            true_traj = body_prop[..., 0:7].clone()
            
            body_prop_s, vel_s, ang_vel_s, true_traj_s = process_stacking_data_dynamic(
                body_prop, true_traj, vel, ang_vel, SEGMENTS=current_segment_len
            )
            
            z0 = torch.cat([
                body_prop_s[:, 0, :, :], 
                vel_s[:, 0, :, :],       
                ang_vel_s[:, 0, :, :]    
            ], dim=-1)
            
            sim_steps = true_traj_s.shape[1]
            t = torch.linspace(0, (sim_steps-1)/25.0, steps=sim_steps, device=device).unsqueeze(0)
            
            optimizer.zero_grad()
            
            pred_traj = model(z0, t)
            #quat_norms = torch.norm(pred_traj[..., 3:7], dim=-1)
            #print(f"Output quat norms: min={quat_norms.min():.6f}, max={quat_norms.max():.6f}")
            pred_pos = pred_traj[..., 0:3]
            pred_quat = pred_traj[..., 3:7]
            
            true_pos = true_traj_s[..., 0:3]
            true_quat = true_traj_s[..., 3:7]

            loss_pos = criterion(pred_pos, true_pos)
            loss_quat = criterion(pred_quat, true_quat)
            loss = loss_pos + loss_quat    # 根据debug的观察，二者还是存在数量级的差距，不知道在val set上表现怎么样，可能得调整一下
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

            # logging.info(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f} | "
            #                  f"(Pos: {loss_pos:.6f}, Quat: {loss_quat:.6f}) ")
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        time_elapsed = time.time() - start_time
        train_history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            # 先运行验证计算损失
            val_loss, val_loss_pos, val_loss_quat = validate_epoch(
                model, val_loader, criterion, device, args, save_predictions=False
            )
            
            train_history['val_loss'].append(val_loss)
            train_history['val_loss_pos'].append(val_loss_pos)
            train_history['val_loss_quat'].append(val_loss_quat)
            
            print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} (Pos: {val_loss_pos:.6f}, Quat: {val_loss_quat:.6f}) | "
                  f"Time: {time_elapsed:.2f}s")
            
            logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, Val Pos: {val_loss_pos:.6f}, "
                        f"Val Quat: {val_loss_quat:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_best.pt'))
                print(f"  --> New best model saved! (Val Loss: {val_loss:.6f})")
                logging.info(f"New best model saved at epoch {epoch+1}")
                
                # 只在最佳模型时保存可视化数据
                if args.save_vis_data:
                    print(f"  --> Collecting visualization data for best model...")
                    # 重新运行验证以收集可视化数据
                    _, _, _, predictions = validate_epoch(
                        model, val_loader, criterion, device, args, save_predictions=True
                    )
                    
                    vis_save_path = os.path.join(vis_dir, f'validation_best_epoch{epoch+1}.npz')
                    np.savez(vis_save_path,
                            stable_scenes=predictions['stable_scenes'],
                            unstable_scenes=predictions['unstable_scenes'],
                            val_stable_indices=val_stable_indices,
                            val_unstable_indices=val_unstable_indices,
                            epoch=epoch+1,
                            val_loss=val_loss,
                            args=vars(args))
                    print(f"  --> Visualization data saved: {vis_save_path}")
                    print(f"      ({len(predictions['stable_scenes'])} stable, {len(predictions['unstable_scenes'])} unstable scenes)")
                
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.6f} | "
                  f"Time: {time_elapsed:.2f}s")
            logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}")
        
        scheduler.step()
        
        # 定期保存checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pt'))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_final.pt'))
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'train_history.npz')
    np.savez(history_path, **train_history)
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Training history saved to: {history_path}")
    logging.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
