#!/bin/bash
# 一键提交 5 组 1-scene overfit 调参实验
# 用法: bash run_5exps.sh

set -e

QUAT_DATA="/mnt/nfs_project_a/chang/small_data/data/blocktower"
EULER_DATA="/mnt/nfs_project_a/chang/data_euler/data_euler/blocktower"

# ============================================================
# Exp 1: quat_baseline — Quaternion 新默认值基线
# ============================================================
mkdir -p exps/quat_baseline
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=quat_base
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/quat_baseline/job.log
#SBATCH --error=exps/quat_baseline/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 1: quat_baseline === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python 1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/small_data/data/blocktower \
    --save_dir exps/quat_baseline \
    --model_name neural_simulator \
    --epochs 500 \
    --lr 1e-3 \
    --eta_min 1e-5 \
    --hidden_dim 256 \
    --layer_num 4 \
    --step_size 0.0025 \
    --dist_boundary 0.02 \
    --weight_decay 1e-5 \
    --quat_loss_weight 0.1 \
    --seed 42

echo "=== Exp 1 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 1: quat_baseline"

# ============================================================
# Exp 2: euler_baseline — Euler 新默认值基线
# ============================================================
mkdir -p exps/euler_baseline
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=euler_base
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/euler_baseline/job.log
#SBATCH --error=exps/euler_baseline/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 2: euler_baseline === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python euler_1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/data_euler/data_euler/blocktower \
    --save_dir exps/euler_baseline \
    --model_name euler_neural_simulator \
    --epochs 500 \
    --lr 1e-3 \
    --eta_min 1e-5 \
    --hidden_dim 256 \
    --layer_num 4 \
    --step_size 0.0025 \
    --dist_boundary 0.02 \
    --weight_decay 1e-5 \
    --euler_loss_weight 0.1 \
    --seed 42

echo "=== Exp 2 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 2: euler_baseline"

# ============================================================
# Exp 3: quat_optuna — Optuna 最优参数 + 新 curriculum
# ============================================================
mkdir -p exps/quat_optuna
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=quat_opt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/quat_optuna/job.log
#SBATCH --error=exps/quat_optuna/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 3: quat_optuna === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python 1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/small_data/data/blocktower \
    --save_dir exps/quat_optuna \
    --model_name neural_simulator \
    --epochs 500 \
    --lr 0.00778 \
    --eta_min 5e-4 \
    --hidden_dim 256 \
    --layer_num 2 \
    --step_size 0.005537 \
    --dist_boundary 0.01555 \
    --weight_decay 0 \
    --quat_loss_weight 0.1 \
    --seed 42

echo "=== Exp 3 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 3: quat_optuna"

# ============================================================
# Exp 4: quat_aggr — 高 lr + 零正则 + 侧重 position
# ============================================================
mkdir -p exps/quat_aggr
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=quat_aggr
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/quat_aggr/job.log
#SBATCH --error=exps/quat_aggr/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 4: quat_aggr === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python 1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/small_data/data/blocktower \
    --save_dir exps/quat_aggr \
    --model_name neural_simulator \
    --epochs 500 \
    --lr 5e-3 \
    --eta_min 1e-4 \
    --hidden_dim 256 \
    --layer_num 4 \
    --step_size 0.0025 \
    --dist_boundary 0.02 \
    --weight_decay 0 \
    --quat_loss_weight 0.05 \
    --seed 42

echo "=== Exp 4 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 4: quat_aggr"

# ============================================================
# Exp 5: euler_high_lr — Euler + 高 lr + 零正则
# ============================================================
mkdir -p exps/euler_high_lr
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=euler_hlr
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/euler_high_lr/job.log
#SBATCH --error=exps/euler_high_lr/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 5: euler_high_lr === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python euler_1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/data_euler/data_euler/blocktower \
    --save_dir exps/euler_high_lr \
    --model_name euler_neural_simulator \
    --epochs 500 \
    --lr 5e-3 \
    --eta_min 1e-4 \
    --hidden_dim 256 \
    --layer_num 4 \
    --step_size 0.0025 \
    --dist_boundary 0.02 \
    --weight_decay 0 \
    --euler_loss_weight 0.1 \
    --seed 42

echo "=== Exp 5 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 5: euler_high_lr"

echo ""
echo "===== All 5 experiments submitted! ====="
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      exps/<name>/train.log"


# ============================================================
# Exp 6: quat_overlap5 — Quaternion overlap stride=5
# ============================================================
mkdir -p exps/quat_overlap5
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=quat_ov5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/quat_overlap5/job.log
#SBATCH --error=exps/quat_overlap5/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 6: quat_overlap5 === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python 1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/small_data/data/blocktower \
    --save_dir exps/quat_overlap5 \
    --model_name neural_simulator \
    --epochs 500 \
    --lr 5e-3 \
    --eta_min 1e-4 \
    --hidden_dim 256 \
    --layer_num 4 \
    --step_size 0.0025 \
    --dist_boundary 0.02 \
    --weight_decay 0 \
    --quat_loss_weight 0.05 \
    --segment_stride 5 \
    --seed 42

echo "=== Exp 6 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 6: quat_overlap5"

# ============================================================
# Exp 7: quat_overlap1 — Quaternion overlap stride=1
# ============================================================
mkdir -p exps/quat_overlap1
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=quat_ov1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/quat_overlap1/job.log
#SBATCH --error=exps/quat_overlap1/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 7: quat_overlap1 === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python 1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/small_data/data/blocktower \
    --save_dir exps/quat_overlap1 \
    --model_name neural_simulator \
    --epochs 500 \
    --lr 5e-3 \
    --eta_min 1e-4 \
    --hidden_dim 256 \
    --layer_num 4 \
    --step_size 0.0025 \
    --dist_boundary 0.02 \
    --weight_decay 0 \
    --quat_loss_weight 0.05 \
    --segment_stride 1 \
    --seed 42

echo "=== Exp 7 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 7: quat_overlap1"

# ============================================================
# Exp 8: euler_overlap5 — Euler overlap stride=5
# ============================================================
mkdir -p exps/euler_overlap5
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=euler_ov5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/euler_overlap5/job.log
#SBATCH --error=exps/euler_overlap5/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 8: euler_overlap5 === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python euler_1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/data_euler/data_euler/blocktower \
    --save_dir exps/euler_overlap5 \
    --model_name euler_neural_simulator \
    --epochs 500 \
    --lr 5e-3 \
    --eta_min 1e-4 \
    --hidden_dim 256 \
    --layer_num 4 \
    --step_size 0.0025 \
    --dist_boundary 0.02 \
    --weight_decay 0 \
    --euler_loss_weight 0.1 \
    --segment_stride 5 \
    --seed 42

echo "=== Exp 8 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 8: euler_overlap5"

# ============================================================
# Exp 9: euler_overlap1 — Euler overlap stride=1
# ============================================================
mkdir -p exps/euler_overlap1
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=euler_ov1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/euler_overlap1/job.log
#SBATCH --error=exps/euler_overlap1/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 9: euler_overlap1 === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python euler_1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/data_euler/data_euler/blocktower \
    --save_dir exps/euler_overlap1 \
    --model_name euler_neural_simulator \
    --epochs 500 \
    --lr 5e-3 \
    --eta_min 1e-4 \
    --hidden_dim 256 \
    --layer_num 4 \
    --step_size 0.0025 \
    --dist_boundary 0.02 \
    --weight_decay 0 \
    --euler_loss_weight 0.1 \
    --segment_stride 1 \
    --seed 42

echo "=== Exp 9 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 9: euler_overlap1"
