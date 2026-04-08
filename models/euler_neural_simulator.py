import torch
import torch.nn as nn
import sys 
from torchdiffeq import odeint , odeint_adjoint

from utils.util import *

FEATURE_DIM = 10 # (x,y,z,rx,ry,rz,lx,ly,lz,dynamic_mask)

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
        # 3: 施力者的几何特征 (lx, ly, lz)，施力者的位置和朝向已经通过相对坐标系变换隐式编码了，所以分支网络只需要处理尺寸特征
        branch_input_dim = 3
        
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
        # 输入是物体的所有几何特征 + 速度 + 角速度，输出是地面力的大小和方向
        # 8(去除了x和y，因为地面应该具有平移不变性)+3+3 = 14维
        ground_input_dim = 8 + 3 + 3
            
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
        q_conj = torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)        
        return self.quat_rotate(q_conj, v)

    def quat_multiply(self, q1, q2):
        """
        Hamilton product q1 * q2
        q1, q2: [..., 4] in (x, y, z, w) format
        """
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        return torch.stack([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dim=-1)

    def euler_to_quat(self, euler):
        """
        Convert Euler angles (rx, ry, rz) to quaternion (x, y, z, w)
        euler: [..., 3] in radians
        """
        rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
        
        cx = torch.cos(rx / 2)
        sx = torch.sin(rx / 2)
        cy = torch.cos(ry / 2)
        sy = torch.sin(ry / 2)
        cz = torch.cos(rz / 2)
        sz = torch.sin(rz / 2)

        q = torch.stack([
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ], dim=-1)

        return torch.nn.functional.normalize(q, p=2, dim=-1)

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
        init_x: [batch, obj_num, 10] - objects applying force (x,y,z,rx,ry,rz,lx,ly,lz,dynamic_mask)
        query_x: [batch, target_obj_num, 10] - objects receiving force (x,y,z,rx,ry,rz,lx,ly,lz,dynamic_mask)
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
        init_x_exp = init_x.unsqueeze(2).expand(-1, -1, target_obj_num, -1)  # [batch, obj_num, target_obj_num, 10]
        query_x_exp = query_x.unsqueeze(1).expand(-1, obj_num, -1, -1)  # [batch, obj_num, target_obj_num, 10]
        init_v_exp = init_v.unsqueeze(2).expand(-1, -1, target_obj_num, -1) # [batch, obj_num, target_obj_num, 3]
        query_v_exp = query_v.unsqueeze(1).expand(-1, obj_num, -1, -1) # [batch, obj_num, target_obj_num, 3]
        init_angular_v_exp = init_angular_v.unsqueeze(2).expand(-1, -1, target_obj_num, -1) # [batch, obj_num, target_obj_num, 3]
        query_angular_v_exp = query_angular_v.unsqueeze(1).expand(-1, obj_num, -1, -1) # [batch, obj_num, target_obj_num, 3]

        # 欧拉角 -> 四元数（仅用于坐标变换和距离计算）
        init_quat = self.euler_to_quat(init_x_exp[..., 3:6])  # [batch, obj_num, target_obj_num, 4]
        query_quat = self.euler_to_quat(query_x_exp[..., 3:6])  # [batch, obj_num, target_obj_num, 4]

        # 构造几何特征: [pos(3), quat(4), size(3), mask(1)] -> 11
        init_geom = torch.cat([init_x_exp[..., 0:3], init_quat, init_x_exp[..., 6:10]], dim=-1)
        query_geom = torch.cat([query_x_exp[..., 0:3], query_quat, query_x_exp[..., 6:10]], dim=-1)

        # 1. 相对位置到局部系
        relative_x_world = (query_geom[...,:3] - init_geom[...,:3]).clone() # [batch, obj_num, target_obj_num, 3]
        relative_x_local = self.quat_rotate_inv(init_quat, relative_x_world) # [batch, obj_num, target_obj_num, 3]

        # 2. 受力者四元数到相对四元数：q_rel = q_init^{-1} * q_query
        init_quat_conj = torch.cat([-init_quat[..., :3], init_quat[..., 3:4]], dim=-1)
        q_rel = self.quat_multiply(init_quat_conj, query_quat) # [batch, obj_num, target_obj_num, 4]
        query_geom_local = torch.cat([
            relative_x_local,
            q_rel,
            query_geom[..., 7:],
        ], dim=-1)

        # 3. 施力者：位置变为原点，四元数为identity
        identity_quat = torch.cat([
            torch.zeros_like(init_quat[..., :3]),
            torch.ones_like(init_quat[..., 3:4])
        ], dim=-1)
        init_geom_local = torch.cat([
            torch.zeros_like(init_geom[...,:3]),
            identity_quat,
            init_geom[..., 7:],
        ], dim=-1)

        # 4. 相对线速度到局部系
        query_v_exp = query_v_exp.clone()
        rel_v_world = query_v_exp - init_v_exp
        query_v_local = self.quat_rotate_inv(init_quat, rel_v_world)

        # 5. 相对角速度到局部系
        rel_angular_v_world = query_angular_v_exp - init_angular_v_exp
        query_angular_v_local = self.quat_rotate_inv(init_quat, rel_angular_v_world)

        # calculate distance (needed by both dist_input and dist_mask independently)
        need_distance = self.use_dist_mask or self.use_dist_input
        if need_distance:
            distance_to_capsule = self.compute_dist_mask(init_geom_local.clone(), query_geom_local.clone())

        # distance mask: zero out forces for far-away pairs
        dist_mask = None
        if self.use_dist_mask:
            boundary = self.dist_boundary
            if scene_scale is not None:
                boundary = self.dist_boundary / scene_scale.view(-1, 1, 1, 1)
            dist_mask = (distance_to_capsule <= boundary).float()

        # distance as input feature to trunk net (independent of mask)
        if self.use_dist_input:
            if dist_mask is not None:
                dist_input = distance_to_capsule * dist_mask * self.dist_input_scale
            else:
                dist_input = distance_to_capsule * self.dist_input_scale

        if self.debug_counter % 200 == 0 and need_distance:
            if dist_mask is not None:
                nonzero_ratio = (dist_mask > 0).float().mean().item()
                print(f"[Mask] nonzero_ratio={nonzero_ratio:.4f} | dist min={distance_to_capsule.min().item():.4f} max={distance_to_capsule.max().item():.4f}", file=sys.stderr)
            else:
                print(f"[Dist] min={distance_to_capsule.min().item():.4f} max={distance_to_capsule.max().item():.4f}", file=sys.stderr)

        # predict force
        branch_input =  torch.cat([init_geom_local[...,7:10]], dim=-1)  # [batch, obj_num, target_obj_num, 3]
        if self.use_dist_input:
            trunk_input =  torch.cat([query_geom_local[...,:10], query_v_local, query_angular_v_local, dist_input], dim=-1)  # [batch, obj_num, target_obj_num, 10+3+3+1]
        else:
            trunk_input =  torch.cat([query_geom_local[...,:10], query_v_local, query_angular_v_local], dim=-1)  # [batch, obj_num, target_obj_num, 10+3+3]
        branch_input_flat = branch_input.reshape(batch_size * obj_num * target_obj_num, branch_input.shape[-1])  # [batch * obj_num * target_obj_num, 3]
        trunk_input_flat = trunk_input.reshape(batch_size * obj_num * target_obj_num, trunk_input.shape[-1])  # [batch * obj_num * target_obj_num, 10+3+3(+1)]

        branch_output = self.branch_net(branch_input_flat)
        trunk_output = self.trunk_net(trunk_input_flat)
        
        # force_flat = self.output_layer(nn.ReLU()(branch_output * trunk_output))
        force_flat = self.output_layer(branch_output * trunk_output)
        force = force_flat.reshape(batch_size, obj_num, target_obj_num, 6)  # [batch, obj_num, target_obj_num, 6]

        if dist_mask is not None:
            force = force * dist_mask

        # 将局部系力转换回世界系
        init_quat_world = self.euler_to_quat(init_x.unsqueeze(2).expand(-1, -1, target_obj_num, -1)[..., 3:6])
        force_linear_world = self.quat_rotate(init_quat_world, force[..., :3])
        force_torque_world = self.quat_rotate(init_quat_world, force[..., 3:6])
        force = torch.cat([force_linear_world, force_torque_world], dim=-1)

        # --- 新增: 预测地面力 ---
        # 不需要两两配对，只需要对每个受力物体(query objects)计算
        # query_x: [batch, target_obj_num, 11]
                
        ground_feat_input = torch.cat([
            query_x[..., 2:10]
        ], dim=-1)
        ground_vel_input = query_v # 速度 [batch, target_obj_num, 3]
        ground_ang_vel_input = query_angular_v # 角速度
        
        ground_input = torch.cat([ground_feat_input, ground_vel_input, ground_ang_vel_input], dim=-1)
        
        ground_force = self.ground_mlp(ground_input) # [batch, target_obj_num, 6]
        
        # 只有在接近地面时才生效? 
        # 可以加个 mask: if z > threshold, ground_force = 0
        # 但通常让神经网络学出来 zero force 更好，或者加上 soft mask
        # 这里加上简单的 soft mask 防止飞很高的物体还受到地面力
        ground_mask = (query_x[..., 2:3] < 1.0).float() # 假设积木不大，0.5m以上不太受地面力
        ground_force = ground_force * ground_mask

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

        if self.debug_counter % 200 == 0:
            pair_mag = force[..., :3].norm(dim=-1).mean().item()
            g_mag = ground_force[..., :3].norm(dim=-1).mean().item()
            print(f"[Force] pair_mean={pair_mag:.4f} | ground_mean={g_mag:.4f}", file=sys.stderr)

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
        pairwise_force = pairwise_force / self.mass
        ground_force = ground_force / self.mass
        
        if self.debug_counter % 200 == 0:
            v_min, v_max = velocities.min().item(), velocities.max().item()
            w_min, w_max = angular_v.min().item(), angular_v.max().item()
            print(f"[InputRange] v=[{v_min:.4f}, {v_max:.4f}] | w=[{w_min:.4f}, {w_max:.4f}]", file=sys.stderr)

        #if self.debug_counter % 200 == 0:
            #gf_z = ground_force[0, :, 2]  # batch=0, all objs, z-force
            #expected = 9.8 / (self.scene_scale[0].item() if self.scene_scale is not None else 1.0)
            #print(f"[Ground] z-force: {gf_z.mean().item():.4f} | expected: {expected:.4f}", file=sys.stderr)


        dxdt = velocities * dynamic_mask
        deuler_dt = angular_v * dynamic_mask

        # 计算线性加速度
        #acceleration = pairwise_force[:,:,:,:3].sum(dim=1)
        # y是高度则是1，z是高度则是2
        #acceleration[:,:,2] += self.gravity
        #acceleration = acceleration * dynamic_mask

        # 计算线性加速度
        acceleration_obj = pairwise_force[:,:,:,:3].sum(dim=1)
        acceleration_ground = ground_force[:,:,:3]
        acceleration = acceleration_obj + acceleration_ground
        if self.scene_scale is not None:
            # 根据场景缩放重调地面重力
            scaled_gravity = self.gravity / self.scene_scale.view(-1, 1) # 场景越大，重力越小（因为距离单位变大了）
            acceleration[:,:,2] += scaled_gravity
        else:
            # y是高度则是1，z是高度则是2
            acceleration[:,:,2] += self.gravity
        
        if self.debug_counter % 200 == 0:
            a_obj = acceleration_obj[..., 2].mean().item()
            a_gnd = acceleration_ground[..., 2].mean().item()
            g = (self.gravity / self.scene_scale.view(-1, 1)).mean().item() if self.scene_scale is not None else self.gravity.item()
            a_net = acceleration[..., 2].mean().item()  # 已包含 gravity
            print(
                f"[Accel-z] obj={a_obj:.4f} | ground={a_gnd:.4f} | gravity={g:.4f} | net_after_g={a_net:.4f}",
                file=sys.stderr
            )        
        acceleration = acceleration * dynamic_mask

        if self.acceleration_clip > 0:
            acceleration = acceleration * (acceleration.abs() > self.acceleration_clip).float() # Deadzone filter
        
        # Safety clamp to prevent explosion
        # acceleration = torch.clamp(acceleration, min=-1e4, max=1e4)

        # 计算角加速度
        # dtheta_scale实际上是转动惯量的倒数的角色？但就是一个常数应该也可以，对于我们所有木块都一样，还是需要根据质量计算？
        # 欧拉方程忽略陀螺项？可以，这是因为我们的场景中通常角速度比较小，不像高速旋转的陀螺那样持续稳定等复杂现象
        dangvdt_obj = pairwise_force[:,:,:,3:6].sum(dim=1) * self.dtheta_scale # [batch_size, obj_num, 3]
        dangvdt_ground = ground_force[:,:,3:6] * self.dtheta_scale # 地面也会产生力矩(摩擦力矩)
        
        dangvdt = dangvdt_obj + dangvdt_ground
        # Safety clamp for angular acceleration
        # dangvdt = torch.clamp(dangvdt, min=-1e4, max=1e4) 
        
        dangvdt = dangvdt * dynamic_mask

        dzdt = torch.cat([
            dxdt,  # derivative of position (3)
            deuler_dt, # derivative of euler angles (3)
            torch.zeros_like(z[:,:,6:9]),  # keep size unchanged (3)
            torch.zeros_like(z[:,:,9:FEATURE_DIM]),  # keep other features (dynamic_mask) unchanged (1)
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
            print(f"  Pos/Euler Deriv NaNs: {torch.isnan(dxdt).sum()} / {torch.isnan(deuler_dt).sum()}", file=sys.stderr)

        if torch.isinf(dzdt).any():
            print("!!! [ODE] Inf detected in dzdt !!!", file=sys.stderr)

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
        return res