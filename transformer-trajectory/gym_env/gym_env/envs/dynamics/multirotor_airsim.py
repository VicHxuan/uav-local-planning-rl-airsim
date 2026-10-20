import airsim
import numpy as np
import math
import time
from gym import spaces

class MultirotorDynamicsAirsim():
    '''
    动力学模型 - 支持 2D (定高) 和 3D (全自由度) 模式及自适应目标游走
    '''
    def __init__(self, cfg) -> None:
        # config
        self.dt = cfg.getfloat('multirotor', 'dt')

        # AirSim Client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Param loading
        self.uav_feature_length = cfg.getint('options', 'uav_feature_length') 
        self.goal_feature_length = cfg.getint('options', 'goal_feature_length') 

        #goal
        self.goal_mode = cfg.get('goal', 'goal_mode') # 'fixed' or 'random'
        self.goal_fixed_x = cfg.getfloat('goal', 'goal_fixed_x')
        self.goal_fixed_y = cfg.getfloat('goal', 'goal_fixed_y')
        self.goal_fixed_z = cfg.getfloat('goal', 'goal_fixed_z')
        self.goal_move_step_length = cfg.getfloat('goal', 'goal_move_step_length')
        self.goal_distance_init = cfg.getfloat('goal', 'goal_distance_init')

        # internal vars
        self.start_position = [0.0, 0.0, 0.0]
        self.goal_position = [0.0, 0.0, 0.0]
        
        # [修改] 游走相关参数
        self.goal_fixed_pose = [self.goal_fixed_x, self.goal_fixed_y, self.goal_fixed_z]
        self.goal_move_step_length = 0.1
        self.goal_move_direction = 0.0     # 2D/3D Azimuth (方位角)
        self.goal_move_elevation = 0.0     # 3D Elevation (俯仰角)
        
        # 物理状态变量 (原始值)
        self.x = 0.0; self.y = 0.0; self.z = 0.0
        self.vx = 0.0; self.vy = 0.0; self.vz = 0.0
        self.ang_vx = 0.0; self.ang_vy = 0.0; self.ang_vz = 0.0
        self.current_distance = 0.0
        self.current_psi_err = 0.0 # 弧度
        
        # 物理状态缓存
        self.state_raw = np.zeros(11, dtype=np.float32) 
        self.state_norm = np.zeros(self.uav_feature_length + self.goal_feature_length) 
        
        self.last_cmd = [0.0, 0.0, 0.0, 0.0]

        # Constraints
        self.acc_xy_max = cfg.getfloat('multirotor', 'acc_xy_max')
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)

        # State Caches for NN
        self.uav_feature_curr = np.zeros(self.uav_feature_length)
        self.goal_feature_curr = np.zeros(self.goal_feature_length)

        # Action Space Logic - Determined by 'dim'
        try:
            self.dim = cfg.getint('options', 'dim')
        except:
            self.dim = 2
        
        if self.dim == 2:
            self.action_cnt = 3 # vx, vy, yaw_rate
        else:
            self.action_cnt = 4 # vx, vy, vz, yaw_rate
            
        print(f"[Dynamics] Dim: {self.dim} -> Action Count: {self.action_cnt}")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_cnt,), dtype=np.float32)

    def reset(self):
        if not hasattr(self, 'goal_distance_init') or self.goal_distance_init is None:
            self.goal_distance_init = 40.0
        self.x, self.y, self.z = float(self.start_position[0]), float(self.start_position[1]), float(self.start_position[2])
        self.last_cmd = [0.0, 0.0, 0.0, 0.0]
        self.state_raw = np.zeros(11, dtype=np.float32)
        print("[DynamicsReset] Internal state cleared")

    def set_action(self, action):
        cmd_vx, cmd_vy, cmd_vz, cmd_yaw_rate = 0.0, 0.0, 0.0, 0.0
        
        self.client.simPause(False)

        if self.dim == 2:
            # === Case 2D: [vx, vy, yaw_rate] ===
            cmd_vx = action[0] * self.v_xy_max
            cmd_vy = action[1] * self.v_xy_max
            cmd_yaw_rate = action[2] * self.yaw_rate_max_rad
            
            # 定高飞行
            target_z = -float(self.start_position[2])
            
            self.client.moveByVelocityZAsync(
                vx=cmd_vx, vy=cmd_vy, z=target_z, duration=self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(cmd_yaw_rate))
            ).join()
            cmd_vz = 0.0

        elif self.dim == 3:
            # === Case 3D: [vx, vy, vz, yaw_rate] ===
            cmd_vx = action[0] * self.v_xy_max
            cmd_vy = action[1] * self.v_xy_max
            cmd_vz = action[2] * self.v_z_max
            cmd_yaw_rate = action[3] * self.yaw_rate_max_rad
            
            self.client.moveByVelocityAsync(
                vx=cmd_vx, vy=cmd_vy, vz=cmd_vz, duration=self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(cmd_yaw_rate))
            ).join()
            
        else:
            print(f"Error: Unsupported dim {self.dim}")
            self.client.simPause(True)
            return

        self.client.simPause(True)
        
        self.last_cmd = [float(cmd_vx), float(cmd_vy), float(cmd_vz), float(cmd_yaw_rate)]

    def init_goal_pose_adaptive(self):
        # ================== 模式 1: 固定目标 ==================
        if self.goal_mode == 'fixed':
            # 直接使用配置文件中的坐标
            gx, gy, gz = self.goal_fixed_pose
            self.goal_position = [float(gx), float(gy), float(gz)]
            
            # 计算一下初始距离，用于后续归一化参考
            dx = gx - self.start_position[0]
            dy = gy - self.start_position[1]
            dz = gz - self.start_position[2]
            self.goal_distance_init = math.sqrt(dx**2 + dy**2 + dz**2) # 更新初始距离参考
            
            print(f"[InitGoal] Fixed Mode | Pos ({gx:.1f}, {gy:.1f}, {gz:.1f})")
            return

        # ================== 模式 2: 随机目标 ==================
        # 1. 确定水平距离和方位角
        self.r = self.goal_distance_init + np.random.uniform(-5, 5)
        phi = np.random.rand() * 2 * math.pi
        
        # 2. 计算 XY 坐标
        dx = self.r * math.cos(phi)
        dy = self.r * math.sin(phi)
        
        # 3. 限制 XY 范围
        limit_xy = 75.0
        goal_x = np.clip(dx, -limit_xy, limit_xy)
        goal_y = np.clip(dy, -limit_xy, limit_xy)
        
        # 4. 计算 Z 坐标 (根据 dim)
        start_z = float(self.start_position[2])
        if self.dim == 2:
            goal_z = start_z
        else:
            # 3D模式: 在起始高度上下 10米 范围内随机，且不低于地面
            z_variation = np.random.uniform(-10.0, 10.0)
            goal_z = np.clip(start_z + z_variation, 2.0, 25.0) 

        self.goal_distance = math.sqrt(dx*dx + dy*dy) 
        self.goal_position = [float(goal_x), float(goal_y), float(goal_z)]
        
        # 初始化游走方向
        self.goal_move_direction = np.random.uniform(0, 2 * math.pi)
        self.goal_move_elevation = np.random.uniform(-math.pi/6, math.pi/6)
        
        print(f"[InitGoal] Random Mode {self.dim}D | Pos ({goal_x:.1f}, {goal_y:.1f}, {goal_z:.1f})")

    def update_goal_pose_adaptive(self):
        """
        [修改] 自适应目标游走更新
        如果是 fixed 模式则不更新；如果是 random 模式则进行随机游走。
        """
        # ================== 模式 1: 固定目标 ==================
        if self.goal_mode == 'fixed':
            return # 固定模式下，目标不移动

        # ================== 模式 2: 随机目标 ==================
        # 使用配置中读取的步长
        base_step = self.goal_move_step_length 
        step_variation = 0.2 * base_step # 步长波动幅度为 20%
        
        direction_drift = math.radians(15) # 方位角漂移
        elevation_drift = math.radians(10) # 俯仰角漂移
        
        actual_step = base_step + np.random.uniform(-step_variation, step_variation)
        
        # 1. 更新方向 (Azimuth)
        direction_offset = np.random.uniform(-direction_drift, direction_drift)
        self.goal_move_direction = (self.goal_move_direction + direction_offset) % (2 * math.pi)
        
        limit_xy = 75.0
        limit_z_min, limit_z_max = 2.0, 25.0 
        
        hit_boundary = False
        
        if self.dim == 2:
            # === 2D 更新逻辑 ===
            dx = actual_step * math.cos(self.goal_move_direction)
            dy = actual_step * math.sin(self.goal_move_direction)
            
            new_x = self.goal_position[0] + dx
            new_y = self.goal_position[1] + dy
            new_z = self.goal_position[2] # Z 不变
            
        else: # dim == 3
            # === 3D 更新逻辑 (球坐标) ===
            # 更新俯仰角
            elev_offset = np.random.uniform(-elevation_drift, elevation_drift)
            self.goal_move_elevation = np.clip(self.goal_move_elevation + elev_offset, -math.pi/2.5, math.pi/2.5)
            
            c_ele = math.cos(self.goal_move_elevation)
            s_ele = math.sin(self.goal_move_elevation)
            
            dx = actual_step * c_ele * math.cos(self.goal_move_direction)
            dy = actual_step * c_ele * math.sin(self.goal_move_direction)
            dz = actual_step * s_ele
            
            new_x = self.goal_position[0] + dx
            new_y = self.goal_position[1] + dy
            new_z = self.goal_position[2] + dz
        
        # 2. 边界检查 (XY)
        if new_x < -limit_xy or new_x > limit_xy:
            new_x = np.clip(new_x, -limit_xy, limit_xy)
            self.goal_move_direction = math.pi - self.goal_move_direction 
            hit_boundary = True
            
        if new_y < -limit_xy or new_y > limit_xy:
            new_y = np.clip(new_y, -limit_xy, limit_xy)
            self.goal_move_direction = -self.goal_move_direction
            hit_boundary = True
            
        # 3. 边界检查 (Z)
        if self.dim == 3:
            if new_z < limit_z_min or new_z > limit_z_max:
                new_z = np.clip(new_z, limit_z_min, limit_z_max)
                self.goal_move_elevation = -self.goal_move_elevation 
        
        # 4. 应用位置
        self.goal_position = [float(new_x), float(new_y), float(new_z)]
        
        # 5. 如果没有撞墙，稍微增加随机扰动
        if not hit_boundary:
            self.goal_move_direction += np.random.uniform(-0.05, 0.05)

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle
    
    def set_goal(self, distance=None, random_angle=0, rect=None):
        if distance is not None: self.goal_distance_init = distance
        self.goal_random_angle = random_angle

    def _update_kinematics(self):
        state = self.client.getMultirotorState()
        kinematics = state.kinematics_estimated
        
        pos = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val])
        self.x, self.y, self.z = pos[0], pos[1], pos[2]
        
        r, p, y = airsim.to_eularian_angles(kinematics.orientation)
        
        self.vx = kinematics.linear_velocity.x_val
        self.vy = kinematics.linear_velocity.y_val
        self.vz = kinematics.linear_velocity.z_val
        
        self.ang_vx = kinematics.angular_velocity.x_val
        self.ang_vy = kinematics.angular_velocity.y_val
        self.ang_vz = kinematics.angular_velocity.z_val

        delta_pos = np.array(self.goal_position) - pos
        self.current_distance = np.linalg.norm(delta_pos) 
        
        target_angle = math.atan2(delta_pos[1], delta_pos[0])
        psi_err = target_angle - y
        while psi_err > math.pi: psi_err -= 2 * math.pi
        while psi_err < -math.pi: psi_err += 2 * math.pi
        self.current_psi_err = psi_err

        self.state_raw = np.array([
            self.x, self.y, self.z,
            r, p, y,
            self.vx, self.vy, self.vz,
            self.current_distance,
            self.current_psi_err
        ], dtype=np.float32)
        
        return pos, (r, p, y), np.array([self.vx, self.vy, self.vz])

    def get_uav_feature_normalized(self):
        _, (r, p, y), _ = self._update_kinematics()
        
        n_vx = np.clip(self.vx / self.v_xy_max, -1.0, 1.0)
        n_vy = np.clip(self.vy / self.v_xy_max, -1.0, 1.0)
        
        norm_factor = max(self.yaw_rate_max_rad, 0.1)
        n_wz = np.clip(self.ang_vz / norm_factor, -1.0, 1.0)
        
        if self.dim == 2:
            uav_feature = np.array([
                n_vx, n_vy,
                math.sin(y), math.cos(y),
                n_wz
            ], dtype=np.float32)
            
        else: # dim == 3
            n_vz = np.clip(self.vz / self.v_z_max, -1.0, 1.0)
            n_wx = np.clip(self.ang_vx / norm_factor, -1.0, 1.0)
            n_wy = np.clip(self.ang_vy / norm_factor, -1.0, 1.0)
            
            uav_feature = np.array([
                n_vx, n_vy, n_vz,
                math.sin(r), math.cos(r),
                math.sin(p), math.cos(p),
                math.sin(y), math.cos(y),
                n_wx, n_wy, n_wz
            ], dtype=np.float32)

        self.uav_feature_curr = uav_feature
        return self.uav_feature_curr

    def get_goal_feature_normalized(self):
        self._update_kinematics()
        pos_current = np.array([self.x, self.y, self.z])
        pos_goal = np.array(self.goal_position)
        
        d_denom = self.goal_distance_init if self.goal_distance_init > 0.1 else 1.0
        
        if self.dim == 2:
            vec_2d = pos_goal[:2] - pos_current[:2]
            dist_2d = np.linalg.norm(vec_2d)
            
            if dist_2d > 0.01:
                norm_dir = vec_2d / dist_2d
            else:
                norm_dir = np.zeros(2)
                
            norm_dist = np.clip(dist_2d / d_denom, 0.0, 1.0)
            self.goal_feature_curr = np.concatenate([norm_dir, [norm_dist]]).astype(np.float32)
            
        else: # dim == 3
            vec_3d = pos_goal - pos_current
            dist_3d = np.linalg.norm(vec_3d)
            
            if dist_3d > 0.01:
                norm_dir = vec_3d / dist_3d
            else:
                norm_dir = np.zeros(3)
                
            norm_dist = np.clip(dist_3d / d_denom, 0.0, 1.0)
            self.goal_feature_curr = np.concatenate([norm_dir, [norm_dist]]).astype(np.float32)
        
        return self.goal_feature_curr, self.current_distance, self.current_psi_err

    def get_current_distance_adaptive(self):
        pos_current = np.array([self.x, self.y, self.z])
        pos_goal = np.array(self.goal_position)
        
        if self.dim == 2:
            return float(np.linalg.norm(pos_goal[:2] - pos_current[:2]))
        else:
            return float(np.linalg.norm(pos_goal - pos_current))

    def get_position(self): self._update_kinematics(); return [self.x, self.y, -self.z]
    def get_velocity(self): return np.array([self.vx, self.vy, self.vz])
    def get_attitude(self): return [self.state_raw[3], self.state_raw[4], self.state_raw[5]]
    
    def get_attitude_cmd(self):
        if not hasattr(self, 'last_cmd'): return [0.0, 0.0, 0.0]
        return [0.0, 0.0, self.last_cmd[3]]
        
    def get_distance_to_goal_2d(self): 
        pos_current = np.array([self.x, self.y, self.z])
        pos_goal = np.array(self.goal_position)
        return float(np.linalg.norm(pos_goal[:2] - pos_current[:2]))
