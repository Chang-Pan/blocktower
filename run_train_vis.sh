#! /bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=start
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=train_vis_cur.log
#SBATCH --error=train_vis_cur.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

# sed -i 's/\r$//' run_train_vis.sh
# === 环境设置 ===
# 加载必要的模块 (根据集群环境修改)
# module load cuda/11.3
# module load anaconda3

# 激活 Python 虚拟环境
# source activate my_physics_env 

# === 运行配置 ===
echo "Job Start Time: $(date)"
echo "Node: $(hostname)"

# 确保在脚本所在目录运行 (即 code 文件夹)
# 如果你不在 code 目录下提交 sbatch，请取消下面一行的注释并修改路径
# cd /your/path/to/project/code

# 创建实验目录
mkdir -p exps/my_exp_cur

# 运行训练脚本
# 调整了参数以适合正式训练：

python train_for_vis_traj.py --dist_boundary 0.03

echo "Job End Time: $(date)"
