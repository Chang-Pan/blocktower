import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset
import os
import glob
from collections import defaultdict

FEATURE_DIM = 11 

class BlockTowerData(Dataset):
    def __init__(self, data_path, max_len=150, scene_type='all'):
        """
        data_path: 文件夹路径
        max_len: 序列最大帧数
        scene_type: 'all' | 'stable' | 'unstable'
        """
        self.data_path = data_path
        self.max_len = max_len
        self.scene_type = scene_type
        
        print(f"[BlockTowerData] Initialized with scene_type='{scene_type}'")

        all_file_paths = glob.glob(os.path.join(data_path, "*.npy"))
        if len(all_file_paths) == 0:
            print(f"Warning: No .npy files found in {data_path}")
        
        # 1. 仅保存文件路径,不加载数据
        self.data_files = [] # 存储 (path, obj_num, filename)
        
        # 用于 Sampler 的索引分组
        self.indices_by_num = defaultdict(list)
        
        print(f"Scanning filenames in {data_path} (Filter: {scene_type})...")

        # 2. 解析文件名建立索引
        valid_count = 0
        for file_path in all_file_paths:
            filename = os.path.basename(file_path)
            
            # --- 过滤逻辑 ---
            if self.scene_type == 'stable' and 'unstable' in filename:
                continue
            if self.scene_type == 'unstable' and 'unstable' not in filename:
                continue
            
            try:
                # 样例: dark_gray_17_stable_0.npy -> ['dark', 'gray', '17', 'stable', '0.npy']
                parts = filename.split('_')
                block_cnt = int(parts[2])
                
                # 物体数量 = 积木数 + 1 (地面)
                obj_num = block_cnt 
                
                # 记录信息
                self.data_files.append((file_path, obj_num, filename))
                self.indices_by_num[obj_num].append(valid_count)
                valid_count += 1
                
            except Exception as e:
                continue

        print(f"Indexing complete. Found {len(self.data_files)} valid files matching '{scene_type}'.")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # === 懒加载 (Lazy Loading) ===
        file_path, obj_num, filename = self.data_files[idx]
        
        try:
            # 读取 .npy
            traj = np.load(file_path).astype(np.float32)

            # 截断 (如果需要)
            if traj.shape[0] > self.max_len:
                traj = traj[:self.max_len]
            
            # 移除地面 (Index 0)
            # data shape: [frames, obj_num_original, features]
            traj = traj[:, 1:, :] 
            
            # 拆分特征
            body_prop = traj[:, :, 0:11]
            vel = traj[:, :, 11:14]
            ang_vel = traj[:, :, 14:17]
            
            # 注意: 这里返回的 obj_num 已经是去掉了地面的数量 (在__init__里改了)
            return filename, body_prop, vel, ang_vel, obj_num
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            dummy_len = self.max_len
            return filename, \
                   np.zeros((dummy_len, obj_num, 11), dtype=np.float32), \
                   np.zeros((dummy_len, obj_num, 3), dtype=np.float32), \
                   np.zeros((dummy_len, obj_num, 3), dtype=np.float32), \
                   obj_num

    def split_train_val(self, val_ratio=0.2, seed=42):
        """
        将数据集划分为训练集和验证集,确保验证集包含相等比例的stable和unstable场景
        
        Args:
            val_ratio: 验证集比例
            seed: 随机种子
            
        Returns:
            train_dataset: 训练集 Subset
            val_dataset: 验证集 Subset
            val_stable_indices: 验证集中stable场景的原始索引
            val_unstable_indices: 验证集中unstable场景的原始索引
        """
        np.random.seed(seed)

        # 分离stable和unstable文件
        stable_indices = [i for i, (_, _, filename) in enumerate(self.data_files) 
                         if 'stable' in filename and 'unstable' not in filename]
        unstable_indices = [i for i, (_, _, filename) in enumerate(self.data_files) 
                           if 'unstable' in filename]

        # 打乱索引
        np.random.shuffle(stable_indices)
        np.random.shuffle(unstable_indices)

        # 确定每个类别的验证集大小
        stable_val_size = int(len(stable_indices) * val_ratio)
        unstable_val_size = int(len(unstable_indices) * val_ratio)

        # 划分训练集和验证集
        stable_val_indices = stable_indices[:stable_val_size]
        stable_train_indices = stable_indices[stable_val_size:]
        unstable_val_indices = unstable_indices[:unstable_val_size]
        unstable_train_indices = unstable_indices[unstable_val_size:]

        # 合并stable和unstable索引
        train_indices = stable_train_indices + unstable_train_indices
        val_indices = stable_val_indices + unstable_val_indices

        # 打乱合并后的索引
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)

        print(f"\n=== Dataset Split Summary ===")
        print(f"Total samples: {len(self)}")
        print(f"Training set: {len(train_dataset)} samples")
        print(f"  - Stable: {len(stable_train_indices)}")
        print(f"  - Unstable: {len(unstable_train_indices)}")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"  - Stable: {len(stable_val_indices)}")
        print(f"  - Unstable: {len(unstable_val_indices)}")
        print(f"============================\n")

        return train_dataset, val_dataset, stable_val_indices, unstable_val_indices

class TrialData(Dataset):
    def __init__(self, data_path, max_len=150, scene_type='all'):
        """
        data_path: 文件夹路径
        max_len: 序列最大帧数
        scene_type: 'all' | 'stable' | 'unstable'
        """
        self.data_path = data_path
        self.max_len = max_len
        self.scene_type = scene_type
        
        print(f"[TrialData] Initialized with scene_type='{scene_type}'")

        all_file_paths = glob.glob(os.path.join(data_path, "*.npy"))
        if len(all_file_paths) == 0:
            print(f"Warning: No .npy files found in {data_path}")
        
        # 1. 仅保存文件路径,不加载数据
        self.data_files = [] # 存储 (path, obj_num, filename)
        
        # 用于 Sampler 的索引分组
        self.indices_by_num = defaultdict(list)
        
        print(f"Scanning filenames in {data_path} (Filter: {scene_type})...")

        # 2. 解析文件名建立索引
        valid_count = 0
        for file_path in all_file_paths:
            filename = os.path.basename(file_path)
            
            # --- 过滤逻辑 ---
            if self.scene_type == 'stable' and 'unstable' in filename:
                continue
            if self.scene_type == 'unstable' and 'unstable' not in filename:
                continue
            
            try:
                # 样例: dark_gray_17_stable_0.npy -> ['dark', 'gray', '17', 'stable', '0.npy']
                parts = filename.split('_')
                block_cnt = int(parts[2])
                
                # 每个config的stable场景中只有前10个是human实验用到的
                if parts[3] == "stable" and int(parts[4].split('.')[0]) >= 10:
                    continue
                
                # 物体数量 = 积木数 + 1 (地面)
                obj_num = block_cnt 
                
                # 记录信息
                self.data_files.append((file_path, obj_num, filename))
                self.indices_by_num[obj_num].append(valid_count)
                valid_count += 1
                
            except Exception as e:
                continue

        print(f"Indexing complete. Found {len(self.data_files)} valid files matching '{scene_type}'.")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # === 懒加载 (Lazy Loading) ===
        file_path, obj_num, filename = self.data_files[idx]
        
        try:
            # 读取 .npy
            traj = np.load(file_path).astype(np.float32)

            # 截断 (如果需要)
            if traj.shape[0] > self.max_len:
                traj = traj[:self.max_len]
            
            # 移除地面 (Index 0)
            # data shape: [frames, obj_num_original, features]
            traj = traj[:, 1:, :] 
            
            # 拆分特征
            body_prop = traj[:, :, 0:11]
            vel = traj[:, :, 11:14]
            ang_vel = traj[:, :, 14:17]
            
            # 注意: 这里返回的 obj_num 已经是去掉了地面的数量 (在__init__里改了)
            return filename, body_prop, vel, ang_vel, obj_num
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            dummy_len = self.max_len
            return filename, \
                   np.zeros((dummy_len, obj_num, 11), dtype=np.float32), \
                   np.zeros((dummy_len, obj_num, 3), dtype=np.float32), \
                   np.zeros((dummy_len, obj_num, 3), dtype=np.float32), \
                   obj_num

class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        支持 Subset 的分组批采样器
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 如果是 Subset,需要访问原始 dataset 的 indices_by_num
        if isinstance(dataset, Subset):
            self.is_subset = True
            self.subset_indices = dataset.indices
            self.original_dataset = dataset.dataset
            # 重建当前 subset 的 indices_by_num
            self.indices_by_num = defaultdict(list)
            for subset_idx, original_idx in enumerate(self.subset_indices):
                _, obj_num, _ = self.original_dataset.data_files[original_idx]
                self.indices_by_num[obj_num].append(subset_idx)
        else:
            self.is_subset = False
            self.indices_by_num = dataset.indices_by_num
        
    def __iter__(self):
        batches = []
        for obj_num, indices in self.indices_by_num.items():
            indices = np.array(indices)
            if self.shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                batches.append(batch.tolist())
        
        if self.shuffle:
            np.random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self):
        count = 0
        for indices in self.indices_by_num.values():
            count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count


def process_stacking_data_dynamic(body_property, true_trajectories, velocity, angular_velocity, SEGMENTS, STRIDE=None):
    """
    适配动态物体数量的切分逻辑
    SEGMENTS: segment_len (片段长度),例如 15
    """
    bs, steps, n_obj, _ = body_property.shape

    segment_len = SEGMENTS
    stride = segment_len if STRIDE is None else STRIDE
    if segment_len <= 0:
        raise ValueError(f"segment_len must be positive, got {segment_len}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if steps < segment_len:
        raise ValueError(f"segment_len={segment_len} is larger than steps={steps}")

    # 生成滑窗起点：默认 stride=segment_len 时与旧逻辑一致
    start_indices = list(range(0, steps - segment_len + 1, stride))
    if len(start_indices) == 0:
        raise ValueError(
            f"No valid segments for steps={steps}, segment_len={segment_len}, stride={stride}"
        )
    
    def split_and_flatten(data):
        segments = [data[:, start:start + segment_len] for start in start_indices]
        d = torch.stack(segments, dim=1)
        return d.reshape(bs * len(start_indices), segment_len, n_obj, -1)

    body_prop_out = split_and_flatten(body_property)
    vel_out = split_and_flatten(velocity)
    ang_vel_out = split_and_flatten(angular_velocity)
    true_traj_out = split_and_flatten(true_trajectories)
    
    return body_prop_out, vel_out, ang_vel_out, true_traj_out

class DebugData(Dataset):
    """
    灵活的调试数据集:
    - 可按 block 数量筛选
    - 可按 stable/unstable 筛选
    - 可指定精确文件名
    - 可选择单场景或多场景
    """
    def __init__(
        self,
        data_path,
        max_len=150,
        single_scene=True,
        block_cnt=None,              # None 表示不过滤；2 表示只要2块
        scene_type='all',            # 'all' | 'stable' | 'unstable'
        target_filename=None,        # 精确文件名优先
        max_scenes=None,             # 限制样本数量；single_scene=True时默认1
        shuffle=False,
        seed=42,
        strict=True                  # 无匹配时是否报错
    ):
        self.data_path = data_path
        self.max_len = max_len
        self.single_scene = single_scene
        self.block_cnt = block_cnt
        self.scene_type = scene_type
        self.target_filename = target_filename

        all_file_paths = glob.glob(os.path.join(data_path, "*.npy"))
        all_file_paths = sorted(all_file_paths)

        candidates = []
        for file_path in all_file_paths:
            filename = os.path.basename(file_path)

            parts = filename.split('_')
            if len(parts) < 5:
                continue

            try:
                curr_block_cnt = int(parts[2])
            except Exception:
                continue

            is_unstable = ('unstable' in filename)
            is_stable = ('stable' in filename and not is_unstable)

            if scene_type == 'stable' and not is_stable:
                continue
            if scene_type == 'unstable' and not is_unstable:
                continue
            if block_cnt is not None and curr_block_cnt != block_cnt:
                continue
            if target_filename is not None and filename != target_filename:
                continue

            candidates.append((file_path, curr_block_cnt, filename))

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(candidates)

        if single_scene and max_scenes is None:
            max_scenes = 1
        if max_scenes is not None:
            candidates = candidates[:max_scenes]

        if len(candidates) == 0:
            msg = (f"[DebugData] No matched file. "
                   f"scene_type={scene_type}, block_cnt={block_cnt}, "
                   f"target_filename={target_filename}")
            if strict:
                raise RuntimeError(msg)
            print(msg)

            # 非 strict 下回退到第一个文件，避免训练直接崩
            if len(all_file_paths) > 0:
                fallback = all_file_paths[0]
                fb_name = os.path.basename(fallback)
                try:
                    fb_cnt = int(fb_name.split('_')[2])
                except Exception:
                    fb_cnt = -1
                candidates = [(fallback, fb_cnt, fb_name)]

        self.data_files = candidates
        self.indices_by_num = defaultdict(list)
        for idx, (_, obj_num, _) in enumerate(self.data_files):
            self.indices_by_num[obj_num].append(idx)

        print(f"[DebugData] Loaded {len(self.data_files)} scene(s)")
        for _, _, name in self.data_files:
            print(f"  - {name}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path, obj_num, filename = self.data_files[idx]
        try:
            traj = np.load(file_path).astype(np.float32)
            if traj.shape[0] > self.max_len:
                traj = traj[:self.max_len]

            traj = traj[:, 1:, :]  # remove ground
            body_prop = traj[:, :, 0:11]
            vel = traj[:, :, 11:14]
            ang_vel = traj[:, :, 14:17]
            return filename, body_prop, vel, ang_vel, obj_num
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            dummy_len = self.max_len
            return filename, \
                   np.zeros((dummy_len, max(obj_num, 1), 11), dtype=np.float32), \
                   np.zeros((dummy_len, max(obj_num, 1), 3), dtype=np.float32), \
                   np.zeros((dummy_len, max(obj_num, 1), 3), dtype=np.float32), \
                   obj_num