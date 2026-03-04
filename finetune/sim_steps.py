import numpy as np
import pandas as pd
import os
import argparse

# 将单步决策逻辑提取为单独的函数
def get_decision_at_step(scene_data, current_sim_step):
    # 确保不越界
    max_steps = scene_data['pred'].shape[0] - 1
    actual_step = min(current_sim_step, max_steps)
    
    # 模拟轨迹片段
    # 如果step包含0，会导致切片为空，这里做个保护，虽然循环通常从1开始
    pred_traj = scene_data['pred'][:actual_step+1]  # [step, obj, 17]
    
    # 保护逻辑：万一长度不足（虽然上面已经min过了）
    if len(pred_traj) < 2:
        # 如果只有1帧（step=0的情况），强制返回稳定，或者使用自身
        original_scene = scene_data['pred'][0]
        final_scene = scene_data['pred'][0]
    else:
        original_scene = pred_traj[0]   # [obj, 17]
        final_scene = pred_traj[-1]     # [obj, 17] 现在这里真的是第 actual_step 帧

    # --- 稳定性判断逻辑 ---
    is_stable = True
    for i in range(pred_traj.shape[1]):
        original_position = original_scene[i][:3]
        final_position = final_scene[i][:3]
            
        z_displacement = np.abs(final_position[2] - original_position[2])
        if z_displacement >= 0.6:
            is_stable = False
            break
    
    if is_stable:
        return "stable"
    
    # --- 颜色倾向判断逻辑 ---
    light_gray = 0
    for i in range(pred_traj.shape[1]):
        if final_scene[i][0] > 0:
            light_gray += 1
    
    if light_gray >= (pred_traj.shape[1]) // 2:
        return "light_gray"
    else:
        return "dark_gray"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=False, default='exps/my_exp/trial_predictions.npy',
                       help='Path to model trial predictions (1800 trials)')
    parser.add_argument('--subject_path', type=str, required=False, default='../data/subject_2/raw_data',
                       help='Path to subject(human) data directory')
    parser.add_argument('--output_dir', type=str, default='exps/my_exp/finetune',
                       help='Output directory')
    # 搜索的最大步数范围
    parser.add_argument('--max_search_steps', type=int, default=150,
                        help="Max simulation steps to search")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Loading predictions from {args.pred_path}...")
    npy_data = np.load(args.pred_path, allow_pickle=True)
    
    # 构建场景名到数据的快速查找字典
    scene_lookup = {}
    for scene in npy_data:
        name = scene['name'].replace('.npy', '')
        scene_lookup[name] = scene

    new_df = []
    
    print(f"Processing subject data from {args.subject_path}...")
    # 遍历被试文件
    for file in os.listdir(args.subject_path):
        # 修正：使用args.subject_path而不是硬编码的路径
        file_path = os.path.join(args.subject_path, file)
        if not os.path.isfile(file_path): continue
        if not file.endswith('.csv'): continue

        df = pd.read_csv(file_path)
        # 兼容性处理
        if 'trajectory_file' in df.columns:
            df = df.drop(['trajectory_file'], axis=1)
        if 'current_acc' in df.columns:
            df = df.drop(['current_acc'], axis=1)

        for i in range(len(df)):
            trial = df.iloc[i].copy()
            stimuli_file = trial['stimuli_file']
            file_info = stimuli_file.split('/')[-1].split('.')[0].split('_')
            scene_name = f"{file_info[0]}_{file_info[1]}_{file_info[2]}_{file_info[3]}_{file_info[4]}"
            
            user_choice = trial['user_choice']
            
            # --- 核心搜索逻辑 ---
            if scene_name not in scene_lookup:
                print(f"Warning: Scene {scene_name} not found in predictions.")
                matching_steps = []
            else:
                scene_data = scene_lookup[scene_name]
                matching_steps = []
                
                # 遍历 1 到 150 步 (或者 args.max_search_steps)
                for step in range(1, args.max_search_steps + 1):
                    pred_decision = get_decision_at_step(scene_data, step)
                    if pred_decision == user_choice:
                        matching_steps.append(step)
            
            # 记录数据
            trial['stimuli_file'] = scene_name
            
            # 将匹配的步数列表转换为分号分隔的字符串，如果为空则为空字符串
            trial['matching_sim_steps'] = ";".join(map(str, matching_steps))
            
            # 如果列表不为空，说明能够 fit human
            if len(matching_steps) > 0:
                trial['can_fit_human'] = 1
            else:
                trial['can_fit_human'] = 0
            
            # 保留原本的 correct_answer 逻辑用于比较
            # 这里不再记录单一的 pred_choice，因为是动态搜索的
            
            new_df.append(trial)
    
    new_df = pd.DataFrame(new_df)
    output_filename = f"{args.output_dir}/finetune_search_steps.csv"
    new_df.to_csv(output_filename, index=False)
    
    print(f"Processing complete.")
    print(f"Total trials processed: {new_df.shape[0]}")
    
    # 计算统计数据
    # can_fit_human 为 1 表示模型至少有一个 step 产生的决策与人一致
    potential_fit_accuracy = new_df['can_fit_human'].sum() / len(new_df)

    # 区分 Stable 和 Unstable 场景的拟合能力
    stable_trials = new_df[new_df['correct_answer'] == 'stable']
    unstable_trials = new_df[new_df['correct_answer'] != 'stable']

    stable_fit_acc = stable_trials['can_fit_human'].mean() if len(stable_trials) > 0 else 0
    unstable_fit_acc = unstable_trials['can_fit_human'].mean() if len(unstable_trials) > 0 else 0

    print("-" * 30)
    print(f"Search Range: 1 to {args.max_search_steps} simulation steps")
    print(f"Overall Potential Human Fit Accuracy: {potential_fit_accuracy:.4f}")
    print(f"(Meaning: For {potential_fit_accuracy*100:.2f}% of trials, there exists at least one simulation step count where the model agrees with humans)")
    print("-" * 30)
    print(f"Potential Fit Accuracy on Stable Ground Truth: {stable_fit_acc:.4f}")
    print(f"Potential Fit Accuracy on Unstable Ground Truth: {unstable_fit_acc:.4f}")
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    main()