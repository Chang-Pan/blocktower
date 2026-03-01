import torch
import torch.nn as nn
import sys 
from torchdiffeq import odeint , odeint_adjoint

from utils.util import *

FEATURE_DIM = 11 # (x,y,z,qx,qy,qz,qw,lx,ly,lz,dynamic_mask)

class ForceFieldPredictor(nn.Module):
    """
    Predict the interaction force between objects
    Input: Initial feature and current position of objects
    Output: Predicted force vector
    """
    def __init__(self, hidden_dim, output_layer, use_dist_mask=True, dist_boundary=0, use_dist_input=True, dist_input_scale=1e2, angle_scale=1e2):
        super(ForceFieldPredictor, self).__init__()# 1. 调用父类 PyTorch Module 的初始化
        # 2. 保存传入的配置参数到 self，供 forward 函数使用
        self.use_dist_mask = use_dist_mask
        self.use_dist_input = use_dist_input
        self.dist_input_scale = dist_input_scale
        self.angle_scale = angle_scale
        self.dist_boundary = dist_boundary
                
        # Debug counter
        self.debug_counter = 0

        # 3. 计算 "Trunk Net" (主干网络) 的输入维度
        if self.use_dist_input:
            # 10(受力者几何特征: x, y, z, quaternion, lx, ly, lz) + 3(受力者速度: vx, vy, vz) + 3(受力者角速度: wx, wy, wz) + 1(显式距离)
            trunk_input_dim = 10+3+3+1
        else:
            trunk_input_dim = 10+3+3

        # 4. 计算 "Branch Net" (分支网络) 的输入维度
        # 10(施力者几何特征): 只关心它是谁、在哪、多大
        branch_input_dim = 10
        
        # 5. 构建 Trunk Net (主干网络)
        trunk_layers = []
        # 5.1 第一层：将特征映射到隐空间 (hidden_dim)
        trunk_layers.append(nn.Linear(trunk_input_dim, hidden_dim)) # x_2,y_2,z_2,quaternion_2,lx_2,ly_2,lz_2  v_x,v_y,v_z  w_x,w_y,w_z
        # 5.2 循环构建中间层
        for _ in range(output_layer):
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
        # 5.3 封装成一个序列模型
        self.trunk_net = nn.Sequential(*trunk_layers)

        # 6. 构建 Branch Net (分支网络) - 结构与 Trunk 类似，处理施力者信息
        branch_layers = []
        branch_layers.append(nn.Linear(branch_input_dim, hidden_dim)) # x_1,y_1,z_1,quaternion_1,lx_1,ly_1,lz_1
        for _ in range(output_layer):
            branch_layers.append(nn.ReLU())
            branch_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net = nn.Sequential(*branch_layers)

        # 7. 构建最终输出层
        # 输入是 branch 和 trunk 的逐元素乘积 (hidden_dim)，输出是力 (6维: Fx, Fy, Fz, Tx, Ty, Tz)
        self.output_layer = nn.Linear(hidden_dim,6)

        # P.s: 如果需要单独建模地面对积木块的力，在这里单独加一个网络
        # 输入是物体的所有几何特征 + 速度 + 角速度 + 显式计算距离？，输出是地面力的大小和方向
        # 10+3+3 = 16维
        if self.use_dist_input:
            ground_input_dim = 10 + 3 + 3 + 1
        else:
            ground_input_dim = 10 + 3 + 3
            
        self.ground_mlp = nn.Sequential(
            nn.Linear(ground_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        )
    
    def quat_rotate(self, q, v):
        """
        Rotate vector v by quaternion q
        q: [..., 4] (x, y, z, w)
        v: [..., 3]
        """
        # Normalize quaternion to prevent numerical drift
        q = torch.nn.functional.normalize(q, p=2, dim=-1)

        q_vec = q[..., :3]
        q_scalar = q[..., 3:4]
        
        # v' = v + 2 * cross(q_v, cross(q_v, v) + q_w * v)
        a = torch.cross(q_vec, v, dim=-1) + q_scalar * v
        b = torch.cross(q_vec, a, dim=-1)
        return v + 2.0 * b

    def quat_rotate_inv(self, q, v):
        """
        Rotate vector v by inverse of quaternion q
        q_inv is (-x, -y, -z, w)
        """
        q_conj = q.clone()
        q_conj[..., :3] = -q_conj[..., :3]
        return self.quat_rotate(q_conj, v)

    def sdf_box(self, p, b):
        """
        Signed distance from point p to box with half-extents b
        p: [..., 3] in box local frame
        b: [..., 3] half-extents
        """
        # q = abs(p) - b
        q = torch.abs(p) - b
        
        # dist = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0)
        # We only care about exterior distance for mask usually, but SDF includes interior (negative)
        # For force field masking, we mostly care about positive distance (separation)
        
        dist_outside = torch.norm(torch.clamp(q, min=0.0), dim=-1, keepdim=True)
        dist_inside = torch.clamp(torch.max(q, dim=-1, keepdim=True)[0], max=0.0)
        
        return dist_outside + dist_inside

    def compute_dist_mask(self, init_x_exp, query_x_exp):
        """ 3D OBB-SDF """
        # 特征维度假设:
        # 0:3 -> pos (x, y, z)
        # 3:7 -> quat (x, y, z, w)
        # 7:10 -> size (lx, ly, lz)
        
        # 1. 解包特征
        pos1 = init_x_exp[..., 0:3]
        quat1 = init_x_exp[..., 3:7]
        size1 = init_x_exp[..., 7:10] / 2.0 # 半长 (Half-extents)
        
        pos2 = query_x_exp[..., 0:3]
        quat2 = query_x_exp[..., 3:7]
        size2 = query_x_exp[..., 7:10] / 2.0 # 半长 (Half-extents)

        # 2. 定义局部采样点 (Box中心为0)
        # 我们需要检测 Box 1 的点在 Box 2 的 SDF 中，以及 Box 2 的点在 Box 1 的 SDF 中
        
        # 构造采样点相对坐标 (归一化到 [-1, 1])
        # 8个顶点 + 6个面中心 + 12个棱中心 = 26个点
        device = init_x_exp.device
        dtype = init_x_exp.dtype
        
        # 8 顶点
        corners = [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ]
        # 6 面中心
        face_centers = [
            [-1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, 1, 0],
            [0, 0, -1], [0, 0, 1]
        ]
        # 12 棱中心
        edge_centers = [
            [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0], # xy plane (z=0)
            [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1], # xz plane (y=0)
            [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1]  # yz plane (x=0)
        ]
        
        signs = torch.tensor(corners + face_centers + edge_centers, device=device, dtype=dtype) # [26, 3]

        # 3. 计算: Box 2 的采样点在 Box 1 的场中
        # 准备 Box 2 的局部采样点
        # size2: [..., 3] -> [..., 1, 3] 广播
        # sample_points2_local: [..., 26, 3]
        sample_points2_local = size2.unsqueeze(-2) * signs 
        
        # 变换 Box 2 的点到世界坐标
        # quat2: [..., 4] -> [..., 1, 4]
        points2_rotated = self.quat_rotate(quat2.unsqueeze(-2), sample_points2_local)
        points2_world = pos2.unsqueeze(-2) + points2_rotated
        
        # 将世界坐标点变换到 Box 1 的局部坐标系
        # 相对位置
        rel_pos_2_in_1 = points2_world - pos1.unsqueeze(-2)
        # 逆旋转 quat1
        points2_in_1 = self.quat_rotate_inv(quat1.unsqueeze(-2), rel_pos_2_in_1)
        
        # 计算这些点到 Box 1 (半长 size1) 的 SDF
        # size1: [..., 3] -> [..., 1, 3]
        dists_2_in_1 = self.sdf_box(points2_in_1, size1.unsqueeze(-2)) # [..., 26, 1]
        min_dist_2_in_1 = torch.min(dists_2_in_1, dim=-2)[0] # 取 26 个点中的最小值 -> [..., 1]

        # 4. 对称计算: Box 1 的采样点在 Box 2 的场中
        sample_points1_local = size1.unsqueeze(-2) * signs
        points1_rotated = self.quat_rotate(quat1.unsqueeze(-2), sample_points1_local)
        points1_world = pos1.unsqueeze(-2) + points1_rotated
        
        rel_pos_1_in_2 = points1_world - pos2.unsqueeze(-2)
        points1_in_2 = self.quat_rotate_inv(quat2.unsqueeze(-2), rel_pos_1_in_2)
        
        dists_1_in_2 = self.sdf_box(points1_in_2, size2.unsqueeze(-2))
        min_dist_1_in_2 = torch.min(dists_1_in_2, dim=-2)[0]

        # 5. 综合距离
        # 取两者最小作为近似最短距离
        distance_to_box = torch.min(min_dist_2_in_1, min_dist_1_in_2)
        
        # DEBUG: Distance Stats
        # Dist Min: 当前场景中靠得最近或穿插最深的两个物体之间的距离，负数代表穿模
        # Dist Max: 最远的物体对距离，通常取决于积木塔的高度或分布范围
        # Dist Mean: 场景中所有物体对距离的平均值，反映整体密集程度
        # 最重要的是Min，非常负表示严重穿模，非常正表示完全没有接触
        #if self.debug_counter % 50 == 0:
            #print(f"[FFP] Dist Min: {distance_to_box.min().item():.4f}, Max: {distance_to_box.max().item():.4f}, Mean: {distance_to_box.mean().item():.4f}", file=sys.stderr)

        return distance_to_box
    
    def forward(self, init_x,query_x,init_v,query_v,init_angular_v,query_angular_v, scene_scale=None):
        """
        init_x: [batch, obj_num, 11] - objects applying force (x,y,z,qx,qy,qz,qw,lx,ly,lz,dynamic_mask)
        query_x: [batch, target_obj_num, 11] - objects receiving force (x,y,z,qx,qy,qz,qw,lx,ly,lz,dynamic_mask)
        init_v: [batch, target_obj_num, 3] - velocities of objects applying force (vx, vy, vz)
        query_v: [batch, obj_num, 3] - velocities of objects receiving force
        init_angular_v: [batch, target_obj_num, 3] - angular velocities of objects applying force (wx, wy, wz)
        query_angular_v: [batch, obj_num, 3] - angular velocities of objects receiving force

        return: [batch, obj_num, target_obj_num, 6] - forces applied to objects receiving force (Fx, Fy, Fz, torque_x, torque_y, torque_z)
        """
        batch_size, obj_num, _ = init_x.shape
        target_obj_num = query_x.shape[1]

        # expand to object pairs
        # 将 N 个施力者和 M 个受力者扩展成 NxM 的矩阵，以便两两配对
        init_x_exp = init_x.unsqueeze(2).expand(-1, -1, target_obj_num, -1)  # [batch, obj_num, target_obj_num, 11]
        query_x_exp = query_x.unsqueeze(1).expand(-1, obj_num, -1, -1)  # [batch, obj_num, target_obj_num, 11]
        init_v_exp = init_v.unsqueeze(2).expand(-1, -1, target_obj_num, -1) # [batch, obj_num, target_obj_num, 3]
        query_v_exp = query_v.unsqueeze(1).expand(-1, obj_num, -1, -1) # [batch, obj_num, target_obj_num, 3]
        init_angular_v_exp = init_angular_v.unsqueeze(2).expand(-1, -1, target_obj_num, -1) # [batch, obj_num, target_obj_num, 3]
        query_angular_v_exp = query_angular_v.unsqueeze(1).expand(-1, obj_num, -1, -1) # [batch, obj_num, target_obj_num, 3]

        # apply relative position
        relative_x = (query_x_exp[...,:3] - init_x_exp[...,:3]).clone() # [batch, obj_num, target_obj_num, 3]
        query_x_exp = query_x_exp.clone()
        query_x_exp[..., :3] = relative_x

        init_x_exp = init_x_exp.clone()
        init_x_exp[..., :3] = 0

        # apply relative velocity
        query_v_exp = query_v_exp.clone()
        query_v_exp -= init_v_exp

        # apply relative angular velocity
        # Transform World Frame Angular Velocity Difference to Local Frame
        # This aligns the rotation axis with the local coordinate system of the 'init' object
        # init_quat: [batch, obj_num, target_obj_num, 4]
        init_quat = init_x_exp[..., 3:7] 
        # Calculate difference in world frame
        rel_angular_v_world = query_angular_v_exp - init_angular_v_exp
        # Rotate into local frame: R_inv * (w_query - w_init)
        query_angular_v_exp = self.quat_rotate_inv(init_quat, rel_angular_v_world)

        # calculate distance mask
        if self.use_dist_mask:
            distance_to_capsule = self.compute_dist_mask(init_x_exp.clone(), query_x_exp.clone())
            
            boundary = self.dist_boundary
            if scene_scale is not None:
                boundary = self.dist_boundary / scene_scale.view(-1, 1, 1, 1) # 根据场景缩放距离边界
            
            dist_mask = (distance_to_capsule <= boundary).float() # [batch, obj_num, target_obj_num, 1] 只有在距离小于等于boundary时才有力
            dist_input = distance_to_capsule * dist_mask
            dist_input *= self.dist_input_scale
        # predict force
        branch_input =  torch.cat([init_x_exp[...,:10]], dim=-1)  # [batch, obj_num, target_obj_num, 10]
        if self.use_dist_input:
            trunk_input =  torch.cat([query_x_exp[...,:10], query_v_exp, query_angular_v_exp, dist_input], dim=-1)  # [batch, obj_num, target_obj_num, 10+3+3+1]
        else:
            trunk_input =  torch.cat([query_x_exp[...,:10], query_v_exp, query_angular_v_exp], dim=-1)  # [batch, obj_num, target_obj_num, 10+3+3]
        branch_input_flat = branch_input.reshape(batch_size * obj_num * target_obj_num, branch_input.shape[-1])  # [batch * obj_num * target_obj_num, 10]
        trunk_input_flat = trunk_input.reshape(batch_size * obj_num * target_obj_num, trunk_input.shape[-1])  # [batch * obj_num * target_obj_num, 10+3+3(+1)]

        branch_output = self.branch_net(branch_input_flat)
        trunk_output = self.trunk_net(trunk_input_flat)
        
        # force_flat = self.output_layer(nn.ReLU()(branch_output * trunk_output))
        force_flat = self.output_layer(branch_output * trunk_output)
        force = force_flat.reshape(batch_size, obj_num, target_obj_num, 6)  # [batch, obj_num, target_obj_num, 6]

        if self.use_dist_mask:
            force = force*dist_mask

        # --- 新增: 预测地面力 ---
        # 不需要两两配对，只需要对每个受力物体(query objects)计算
        # query_x: [batch, target_obj_num, 11]
        
        # 1. 计算物体到地面的距离 (简单的 z 坐标, 假设地面是 z=0)
        # 物体坐标是质心，我们需要考虑到底面的距离: z - lz/2
        # 注意: 这里的物体朝向如果不是平的，最低点计算会复杂。
        # 简化假设: 主要用于长方体平放堆叠，用 z - lz/2 近似，或者让神经网络自己通过 (z, quat, size) 去学。
        # 为了让网络容易学，我们构造一个显式的 "height_above_ground" 特征
        
        obj_z = query_x[..., 2:3] # z
        obj_lz = query_x[..., 9:10] # lz (full length in z-axis local?) -> 假设特征 7,8,9 是 lx, ly, lz
        # 注意：你在 compute_dist_mask 里是用 query_x[..., 7:10] 作为 size。
        # 这里为了保持一致，我们直接把所有几何特征扔进去，但额外构造一个 '离地特征'
        
        # 简单粗暴：如果 use_dist_input，我们传入 z 作为距离特征
        # 因为地面在 z=0，距离就是 z (如果可以穿透则可能是负数，不取abs)
        # 或者更精细一点：z - box_height_radius
        # 暂时直接使用 z 坐标作为距离提示，网络会学到 offset
        
        ground_feat_input = query_x[..., :10] # 几何特征
        ground_vel_input = query_v # 速度 [batch, target_obj_num, 3]
        ground_ang_vel_input = query_angular_v # 角速度
        
        if self.use_dist_input:
            # 缩放一下 z，使其在接触面附近数值较大/敏感
            # 这里简单地把 z * scale 作为距离输入
            dist_to_ground = query_x[..., 2:3] * self.dist_input_scale
            ground_input = torch.cat([ground_feat_input, ground_vel_input, ground_ang_vel_input, dist_to_ground], dim=-1)
        else:
            ground_input = torch.cat([ground_feat_input, ground_vel_input, ground_ang_vel_input], dim=-1)
            
        ground_force = self.ground_mlp(ground_input) # [batch, target_obj_num, 6]
        
        # 只有在接近地面时才生效? 
        # 可以加个 mask: if z > threshold, ground_force = 0
        # 但通常让神经网络学出来 zero force 更好，或者加上 soft mask
        # 这里加上简单的 soft mask 防止飞很高的物体还受到地面力
        #ground_mask = (query_x[..., 2:3] < 1.0).float() # 假设积木不大，0.5m以上不太受地面力
        #ground_force = ground_force * ground_mask

        # DEBUG: Force Stats
        self.debug_counter += 1
        #if self.debug_counter % 50 == 0:
            # force has shape [batch, N_senders, N_receivers, 6]
            # We want to see the total force received by each object (Sum over senders)
            #total_force_received = force.sum(dim=1) # [batch, N_receivers, 6]
            #force_mag = total_force_received.norm(dim=-1)
            
            # Print stats for Pairwise forces (original) to see individual interactions
            #pairwise_mag = force.norm(dim=-1)
            
            #print(f"[FFP] Pairwise Max: {pairwise_mag.max().item():.4f} | Total Received Max: {force_mag.max().item():.4f} | Total Mean: {force_mag.mean().item():.4f}", file=sys.stderr)
            
        if torch.isnan(force).any():
            print("!!! [FFP] NaN detected in Force !!!", file=sys.stderr)

        return force, ground_force

class ODEFunc(nn.Module):
    """
    Define the dynamic equation of the system
    - Force field prediction
    - Acceleration calculation
    - State derivative calculation
    """

    def __init__(self, force_predictor, mass=1, dtheta_scale=1,acceleration_clip=0):
        super(ODEFunc, self).__init__()
        self.force_predictor = force_predictor
        self.mass = mass
        self.dtheta_scale = dtheta_scale
        self.acceleration_clip = acceleration_clip
        # 旧值 1/60 (0.016) 极小，可能是针对 frame-based 时间系统的旧设定？
        # 3D 物理中，如果是标准单位 (米, 秒)，g 应为 9.8。
        # 这里默认设为 9.8，如果发现物体下落过快/过慢，请检查空间坐标的单位 (例如是否是厘米) 或时间步长。
        self.gravity = torch.tensor(-9.8, device=self.force_predictor.trunk_net[0].weight.device)
        # self.gravity = torch.tensor(1/60, device=self.force_predictor.trunk_net[0].weight.device) # g=1/60  1/60 * 150 * 15 * mass = 3.75
        
        self.debug_counter = 0
        self.scene_scale = None # 这个值需要在外部根据场景动态设置，或者通过训练数据统计得到一个平均值

    def set_scale(self, scale):
        self.scene_scale = scale

    def forward(self, t, z):
        """
        Input:
            t: current time
            z: state [batch_size, obj_num, FEATURE_DIM + 6] 11 features + 3 velocities + 3 angular velocity
        Output:
            dzdt: state derivative [batch_size, obj_num, FEATURE_DIM + 6]
        """
        batch_size, obj_num, _ = z.shape 

        velocities = z[:, :, FEATURE_DIM:FEATURE_DIM+3]
        angular_v = z[:, :, FEATURE_DIM+3:FEATURE_DIM+6]
        dynamic_mask = z[:, :, FEATURE_DIM-1:FEATURE_DIM]

        all_features = z[:,:,:FEATURE_DIM] # [batch_size, obj_num, FEATURE_DIM]

        # ===== 数值稳定性处理 =====
        # 1. 归一化四元数 (防止ODE积分过程中偏离单位长度)
        curr_quat = all_features[:, :, 3:7]
        quat_norm = torch.norm(curr_quat, dim=-1, keepdim=True)
        quat_norm = torch.clamp(quat_norm, min=1e-8)  # 防止除以零
        curr_quat_normalized = curr_quat / quat_norm
        all_features = all_features.clone()  # 避免就地修改
        all_features[:, :, 3:7] = curr_quat_normalized
        
        # 2. 钳制速度和角速度 (防止数值爆炸)
        velocities = torch.clamp(velocities, min=-100.0, max=100.0)
        angular_v = torch.clamp(angular_v, min=-100.0, max=100.0)

        # 在归一化后
        # if t.item() == 0 and torch.rand(1).item() < 0.01:  # 1%概率打印
        #     print(f"[Debug] Quat norm before: {torch.norm(curr_quat, dim=-1).mean():.6f}")
        #     print(f"[Debug] Quat norm after: {torch.norm(curr_quat_normalized, dim=-1).mean():.6f}")
        #     print(f"[Debug] Pos norm before: {torch.norm(curr_pos, dim=-1).mean():.6f}")
        #     print(f"[Debug] Pos norm after: {torch.norm(curr_pos_normalized, dim=-1).mean():.6f}")
        #     print(f"[Debug] Velocity range: [{velocities.min():.2f}, {velocities.max():.2f}]")

        pairwise_force, ground_force = self.force_predictor(
            init_x=all_features,
            query_x=all_features,
            init_v=velocities,
            query_v=velocities,
            init_angular_v=angular_v,
            query_angular_v=angular_v,
            scene_scale=self.scene_scale
        ) # [batch_size, obj_num, obj_num, 6] [i,j] means the force of object j received from object i
        # ground_force: [batch, obj_num, 6]

        # mask掉自己对自己的力
        mask = 1- torch.eye(obj_num, device=z.device).unsqueeze(0) # [1, obj_num, obj_num]
        mask = mask.unsqueeze(-1) # [1, obj_num, obj_num, 1]
        pairwise_force = pairwise_force * mask # [batch_size, obj_num, obj_num, 6]
        pairwise_force /= self.mass

        # 计算导数，旋转的部分使用四元数导数公式 dq/dt = 0.5 * Omega * q
        # 假设 angular_v 是世界坐标系下的角速度
        
        # 1. 提取当前四元数 (x, y, z, w)
        curr_quat = all_features[:, :, 3:7] 
        qx = curr_quat[..., 0]
        qy = curr_quat[..., 1]
        qz = curr_quat[..., 2]
        qw = curr_quat[..., 3]
        
        # 2. 提取当前角速度 (wx, wy, wz)
        wx = angular_v[..., 0]
        wy = angular_v[..., 1]
        wz = angular_v[..., 2]
        
        # 3. 计算四元数导数 (四元数乘法展开: Omega_world * q)
        # d_q_x =  wx*qw + wy*qz - wz*qy
        # d_q_y = -wx*qz + wy*qw + wz*qx
        # d_q_z =  wx*qy - wy*qx + wz*qw
        # d_q_w = -wx*qx - wy*qy - wz*qz
        dq_x =  wx*qw + wy*qz - wz*qy
        dq_y = -wx*qz + wy*qw + wz*qx
        dq_z =  wx*qy - wy*qx + wz*qw
        dq_w = -wx*qx - wy*qy - wz*qz
        
        dquat_dt = 0.5 * torch.stack([dq_x, dq_y, dq_z, dq_w], dim=-1)

        dxdt = velocities * dynamic_mask
        dquat_dt = dquat_dt * dynamic_mask

        # 计算线性加速度
        #acceleration = pairwise_force[:,:,:,:3].sum(dim=1)
        # y是高度则是1，z是高度则是2
        #acceleration[:,:,2] += self.gravity
        #acceleration = acceleration * dynamic_mask

        # 计算线性加速度
        acceleration_obj = pairwise_force[:,:,:,:3].sum(dim=1)
        acceleration_ground = ground_force[:,:,:3] / self.mass
        acceleration = acceleration_obj + acceleration_ground
        
        if self.scene_scale is not None:
            # 根据场景缩放重调地面重力
            scaled_gravity = self.gravity / self.scene_scale.view(-1, 1) # 场景越大，重力越小（因为距离单位变大了）
            acceleration[:,:,2] += scaled_gravity
        else:
            # y是高度则是1，z是高度则是2
            acceleration[:,:,2] += self.gravity
        
        acceleration = acceleration * dynamic_mask

        if self.acceleration_clip > 0:
            acceleration = acceleration * (acceleration.abs() > self.acceleration_clip).float() # Deadzone filter
        
        # Safety clamp to prevent explosion
        acceleration = torch.clamp(acceleration, min=-1e4, max=1e4)

        # 计算角加速度
        # dtheta_scale实际上是转动惯量的倒数的角色？但就是一个常数应该也可以，对于我们所有木块都一样，还是需要根据质量计算？
        # 欧拉方程忽略陀螺项？可以，这是因为我们的场景中通常角速度比较小，不像高速旋转的陀螺那样持续稳定等复杂现象
        dangvdt_obj = pairwise_force[:,:,:,3:6].sum(dim=1) * self.dtheta_scale # [batch_size, obj_num, 3]
        dangvdt_ground = ground_force[:,:,3:6] * self.dtheta_scale # 地面也会产生力矩(摩擦力矩)
        
        dangvdt = dangvdt_obj + dangvdt_ground
        # Safety clamp for angular acceleration
        dangvdt = torch.clamp(dangvdt, min=-1e4, max=1e4) 
        
        dangvdt = dangvdt * dynamic_mask

        dzdt = torch.cat([
            dxdt,  # derivative of position (3)
            dquat_dt,  # derivative of quaternion (4)
            torch.zeros_like(z[:,:,7:10]),  # keep size unchanged (3)
            torch.zeros_like(z[:,:,10:FEATURE_DIM]),  # keep other features (dynamic_mask) unchanged (1)
            acceleration, # derivative of velocity (3)
            dangvdt, # derivative of angular velocity (3)，与torque保持一致维度
        ], dim=-1)

        # DEBUG: Derivatives
        self.debug_counter += 1
        condition = (self.debug_counter % 50 == 0) or torch.isnan(dzdt).any() or (acceleration.abs().max() > 1e3) or (dangvdt.abs().max() > 1e3)
        # if condition:
        # #     # Calculate Contact Force Acceleration (without gravity) for inspection
        # #     # Accel_contact = F_contact / m
        # #     # We recover it from acceleration (Accel_total) by subtracting gravity
        # #     # acceleration = Accel_contact + g
        # #     # So Accel_contact = acceleration - g
        #     accel_contact = acceleration.clone()
        #     accel_contact[:,:,2] -= self.gravity 
            
        #     print(f"[ODE t={t.item():.3f}] Accel Max: {acceleration.abs().max().item():.2f}, AngAcc Max: {dangvdt.abs().max().item():.2f}", file=sys.stderr)
            
        #     # Print detailed stats for first 5 objects
        #     # Assuming batch 0
        #     n_print = min(5, obj_num)
        #     print(f"  --- Top {n_print} Objects Stats (Batch 0) ---", file=sys.stderr)
        #     for i in range(n_print):
        #         # Z-axis is usually the most interesting for stability (index 2)
        #         # Formatter: [Obj ID] Total_Az(Net) = Contact_Az + Gravity
        #         net_az = acceleration[0, i, 2].item()
        #         contact_az = accel_contact[0, i, 2].item()
        #         grav = (self.gravity).item()
                
        #         print(f"  [Obj {i}] Net_Az: {net_az:6.3f} | Contact_Az: {contact_az:6.3f} (G={grav:.3f}) | Net_Ax: {acceleration[0,i,0].item():.3f}, Net_Ay: {acceleration[0,i,1].item():.3f}", file=sys.stderr)

        if torch.isnan(dzdt).any():
            print("!!! [ODE] NaN detected in dzdt !!!", file=sys.stderr)
            print(f"  Accel NaNs: {torch.isnan(acceleration).sum()}", file=sys.stderr)
            print(f"  AngAcc NaNs: {torch.isnan(dangvdt).sum()}", file=sys.stderr)
            print(f"  Pos/Quat Deriv NaNs: {torch.isnan(dxdt).sum()} / {torch.isnan(dquat_dt).sum()}", file=sys.stderr)

        return dzdt

class NeuralODEModel(nn.Module):
    def __init__(self, ode_func, use_adjoint=False, step_size=1/1200):
        super(NeuralODEModel, self).__init__()
        self.ode_func = ode_func
        self.use_adjoint = use_adjoint
        self.step_size = step_size
    
    def forward(self, z0, t, scene_scale=None):
        # z0: [batch_size, obj_num, FEATURE_DIM+6]
        # t: [batch_size,time_steps]
        # return: [time_steps, batch_size, obj_num, FEATURE_DIM+6]

        self.ode_func.set_scale(scene_scale)
        # 取第一个 batch 的时间序列，通常大家时间是一样的
        t = t[0,:] # [time_steps]

        ts = t.shape[0] # 总时间步数
        # 使用 torchdiffeq 求解 ODE
        method = 'rk4' # Use RK4 for better stability (was 'euler')
        if self.use_adjoint:
            res = odeint_adjoint(self.ode_func, z0, t,method=method,options={'step_size':self.step_size})
        else:
            res = odeint(self.ode_func, z0, t,method=method,options={'step_size':self.step_size})  # [time_steps, batch_size, obj_num, FEATURE_DIM+6]
        res = res.permute(1,0,2,3) # [batch_size, time_steps, obj_num, FEATURE_DIM+6]
        # ===== 输出后处理: 归一化四元数 =====
        # 确保输出的四元数是归一化的
        quat = res[..., 3:7]  # [batch, time, obj, 4]
        quat_norm = torch.norm(quat, dim=-1, keepdim=True)
        quat_norm = torch.clamp(quat_norm, min=1e-8)
        res = res.clone()
        res[..., 3:7] = quat / quat_norm
        return res