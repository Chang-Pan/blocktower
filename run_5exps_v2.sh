#!/bin/bash
# 第二轮实验：新算法改进 vs 第一轮（相同超参，直接A/B对比）
# 改进：sign-aware quat loss, soft sigmoid masks, time-weighted pos loss
# 用法: bash run_5exps_v2.sh
# 对比: exps/*_v2/ vs exps/*/（同名去掉_v2就是第一轮对照组）

set -e

# ============================================================
# Exp 1v2: quat_baseline_v2 — 对照 quat_baseline
# ============================================================
mkdir -p exps/quat_baseline_v2
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=quat_bv2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/quat_baseline_v2/job.log
#SBATCH --error=exps/quat_baseline_v2/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 1v2: quat_baseline_v2 === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python 1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/small_data/data/blocktower \
    --save_dir exps/quat_baseline_v2 \
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

echo "=== Exp 1v2 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 1v2: quat_baseline_v2"

# ============================================================
# Exp 2v2: euler_baseline_v2 — 对照 euler_baseline
# ============================================================
mkdir -p exps/euler_baseline_v2
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=euler_bv2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/euler_baseline_v2/job.log
#SBATCH --error=exps/euler_baseline_v2/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 2v2: euler_baseline_v2 === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python euler_1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/data_euler/data_euler/blocktower \
    --save_dir exps/euler_baseline_v2 \
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

echo "=== Exp 2v2 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 2v2: euler_baseline_v2"

# ============================================================
# Exp 3v2: quat_optuna_v2 — 对照 quat_optuna
# ============================================================
mkdir -p exps/quat_optuna_v2
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=quat_ov2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/quat_optuna_v2/job.log
#SBATCH --error=exps/quat_optuna_v2/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 3v2: quat_optuna_v2 === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python 1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/small_data/data/blocktower \
    --save_dir exps/quat_optuna_v2 \
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

echo "=== Exp 3v2 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 3v2: quat_optuna_v2"

# ============================================================
# Exp 4v2: quat_aggr_v2 — 对照 quat_aggr
# ============================================================
mkdir -p exps/quat_aggr_v2
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=quat_av2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/quat_aggr_v2/job.log
#SBATCH --error=exps/quat_aggr_v2/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 4v2: quat_aggr_v2 === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python 1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/small_data/data/blocktower \
    --save_dir exps/quat_aggr_v2 \
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

echo "=== Exp 4v2 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 4v2: quat_aggr_v2"

# ============================================================
# Exp 5v2: euler_high_lr_v2 — 对照 euler_high_lr
# ============================================================
mkdir -p exps/euler_high_lr_v2
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --job-name=euler_hv2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --output=exps/euler_high_lr_v2/job.log
#SBATCH --error=exps/euler_high_lr_v2/job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=panchang@stu.pku.edu.cn

echo "=== Exp 5v2: euler_high_lr_v2 === $(date)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

python euler_1scene_posnormed_train.py \
    --data_path /mnt/nfs_project_a/chang/data_euler/data_euler/blocktower \
    --save_dir exps/euler_high_lr_v2 \
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

echo "=== Exp 5v2 Done === $(date)"
SBATCH_EOF
echo "[Submitted] Exp 5v2: euler_high_lr_v2"

echo ""
echo "===== Round 2: All 5 experiments submitted! ====="
echo "Compare v2 (new algo) vs v1 (old algo) with same hyperparams"
echo "Monitor: squeue -u \$USER"
echo "Logs:    exps/<name>_v2/train.log"
