import numpy as np
import pandas as pd
import os
import argparse


def get_decision_from_npy(file, sim_steps):
    data = np.load(file, allow_pickle=True)
    decisions = {}
    sim_steps = min(sim_steps, data[0]['pred'].shape[0]-1)
    for scene in data:
        scene_name = scene['name'].replace('.npy', '')
        pred_traj = scene['pred'][:sim_steps]  if sim_steps > 0 else np.array([scene['pred'][:sim_steps]]) # [time, obj, 17]
        original_scene = pred_traj[0]   # [obj, 17]
        final_scene = pred_traj[-1]     # [obj, 17]

        is_stable = True
        for i in range(pred_traj.shape[1]):
            original_position = original_scene[i][:3]
            if sim_steps < 40:
                final_position = scene['pred'][39][i][:3]
            else:
                final_position = final_scene[i][:3]
            z_displacement = np.abs(final_position[2] - original_position[2])
            if z_displacement >= 0.6:
                is_stable = False
                break
        
        if is_stable:
            decisions[scene_name] = "stable"
            continue
        
        light_gray = 0
        for i in range(pred_traj.shape[1]):
            if final_scene[i][0] > 0:
                light_gray += 1
        
        if light_gray >= (pred_traj.shape[1]) // 2:
            decisions[scene_name] = "light_gray"
        else:
            decisions[scene_name] = "dark_gray"
    return decisions


# ！！！注意pred_path是evaluate.py输出的模型预测结果的存储路径，不是原始场景数据
# ！！！注意subject_path是存储被试实验数据的路径，要精确到raw_data
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=False, default='exps/my_exp/trial_predictions.npy',
                       help='Path to model trial predictions (1800 trials)')
    parser.add_argument('--subject_path', type=str, required=False, default='../data/subject_2/raw_data',
                       help='Path to subject(human) data (1800 trials)')
    parser.add_argument('--output_dir', type=str, default='exps/my_exp/finetune',
                       help='Output directory for animations')
    parser.add_argument('--sim_steps', type=int, default=149,
                        help="Simulation steps (clamp)")
    parser.add_argument('--fps', type=float, default=25,
                       help='Animation frame rate')
    args = parser.parse_args()
    decisions = get_decision_from_npy(args.pred_path, args.sim_steps)
    new_df = []
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 被试的实验数据存在多个csv文件中，每个文件是一个session（即线下参加一次）
    for file in os.listdir(args.subject_path):
        df = pd.read_csv(os.path.join("../data/subject_2/raw_data", file))
        df = df.drop(['trajectory_file', 'current_acc'], axis=1)
        for i in range(len(df)):
            trial = df.iloc[i].copy()   # 避免直接在原数据上改
            stimuli_file = trial['stimuli_file']
            file_info = stimuli_file.split('/')[-1].split('.')[0].split('_')
            scene_name = f"{file_info[0]}_{file_info[1]}_{file_info[2]}_{file_info[3]}_{file_info[4]}"  # 形如：[light, gray, 17, unstable, 0]
            pred_decision = decisions[scene_name]
            user_choice = trial['user_choice']
            correct_answer = trial['correct_answer']

            trial['stimuli_file'] = scene_name
            trial['pred_choice'] = pred_decision
            if pred_decision == user_choice:
                trial['model_human'] = 1
            else:
                trial['model_human'] = 0
            
            if pred_decision == correct_answer:
                trial['model_gt'] = 1
            else:
                trial['model_gt'] = 0
            
            new_df.append(trial)
    
    new_df = pd.DataFrame(new_df)
    print(f"Finish {args.sim_steps} steps decision making.")
    print(f"Get {new_df.shape[0]} trials.")
    new_df.to_csv(f"{args.output_dir}/finetune_{args.sim_steps}.csv", index=False)


    model_human_accuracy = new_df['model_human'].sum() / len(new_df)
    model_gt_accuracy = new_df['model_gt'].sum() / len(new_df)


    stable_accuracy = []
    unstable_accuracy = []
    for i in range(len(new_df)):
        trial = new_df.iloc[i]
        if trial['correct_answer'] == "stable":
            stable_accuracy.append(trial['model_human'])
        else:
            unstable_accuracy.append(trial['model_human'])

    print(f"仿真帧数{args.sim_steps} 积分步长1/200")
    print(f"Fit human accuracy: {model_human_accuracy}")
    print(f"Fit gt accuracy: {model_gt_accuracy}")
    print(f"Fit human Stable acuracy: {np.sum(stable_accuracy)/len(stable_accuracy)}")
    print(f"Fit human Unstable accuracy: {np.sum(unstable_accuracy)/len(unstable_accuracy)}")

if __name__ == "__main__":
    main()