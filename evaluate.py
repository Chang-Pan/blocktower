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

from utils.blocktower_data_nff import TrialData, GroupedBatchSampler, process_stacking_data_dynamic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_data_path', type=str, default='/mnt/nfs_project_a/chang/data/data/blocktower', help='Path to the dataset folder containing .npy files')
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

def run_trial_set(model, data_loader, criterion, device, args):
    """
    运行Trial Set，仅保存预测结果用于后续分析，不保存可视化动图数据
    return: List of dicts (name, pred, true)
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (game_names, body_prop, vel, ang_vel, body_nums) in enumerate(data_loader):
            body_prop = body_prop.to(device)
            vel = vel.to(device)
            ang_vel = ang_vel.to(device)
            
            # [新增] 归一化
            pos_initial = body_prop[:, 0, :, 0:3] # [Batch, Obj, 3]
            pos_flat = pos_initial.reshape(body_prop.size(0), -1)
            scene_scale = torch.max(torch.abs(pos_flat), dim=1)[0]
            scene_scale = torch.clamp(scene_scale, min=1.0)
            scale_view = scene_scale.view(-1, 1, 1, 1)

            # 备份原始真值
            true_traj_orig_full = torch.cat([body_prop, vel, ang_vel], dim=-1).clone()

            body_prop[..., 0:3] /= scale_view
            body_prop[..., 7:10] /= scale_view
            vel /= scale_view

            # 第0帧作为初始状态
            z0 = torch.cat([
                body_prop[:, 0, :, :], 
                vel[:, 0, :, :],       
                ang_vel[:, 0, :, :]    
            ], dim=-1)
            
            sim_steps = body_prop.shape[1] # 150
            t = torch.linspace(0, (sim_steps-1)/25.0, steps=sim_steps, device=device).unsqueeze(0)
            
            pred_traj = model(z0, t, scene_scale=scene_scale)  # [batch, time, obj, 17]

            # 反归一化并保存
            pred_traj_np = pred_traj.cpu().numpy()  # [batch, time, obj, 17]
            scale_np = scene_scale.cpu().numpy()  # [batch]
            true_traj_np = true_traj_orig_full.cpu().numpy() # [Batch, Time, Obj, 17]
            
            scale_np_view = scale_np[:, None, None, None]
            pred_traj_np[..., 0:3] *= scale_np_view   # Position
            pred_traj_np[..., 7:10] *= scale_np_view # Size
            pred_traj_np[..., 11:14] *= scale_np_view  # Velocity
            
            batch_size = len(game_names)
            for i in range(batch_size):
                scene_name = game_names[i]
                scene_data = {
                    'name': scene_name,
                    'pred': pred_traj_np[i],      # [time, obj, 17]
                    'true': true_traj_np[i],    # [time, obj, 17]
                    'num_objs': body_nums[i] if isinstance(body_nums, (list, np.ndarray)) else body_nums
                }
                all_predictions.append(scene_data)
    
    return all_predictions


# ！！！注意args.trial_data_path是存放原来的3000个场景的Path，因为实验用到的1800个trial包含在里面，不要设置成train的时候用的小场景path
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
        import models.posnormed_neural_simulator as model_module
        
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
    trial_set = TrialData(data_path=args.trial_data_path, max_len=150, scene_type=args.scene_type)

    # 创建实验试次集DataLoader
    trial_batch_sampler = GroupedBatchSampler(trial_set, batch_size=args.batch_size, shuffle=False)
    trial_loader = DataLoader(trial_set, batch_sampler=trial_batch_sampler, num_workers=0)
    
    criterion = nn.MSELoss()

    state_dict = torch.load(os.path.join(args.save_dir, 'model_best.pt'), map_location=device)
    model.load_state_dict(state_dict)
    trial_predictions = run_trial_set(model, trial_loader, criterion, device, args)
    np.save(f"{args.save_dir}/trial_predictions_step150.npy", trial_predictions)
    print(f"Trial predictions saved to {args.save_dir}/trial_predictions_step150.npy")

if __name__ == "__main__":
    main()
