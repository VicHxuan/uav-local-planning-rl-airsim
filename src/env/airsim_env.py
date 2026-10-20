"""AirSim Gym environment for mapless UAV local planning (env logic plus Qt signals for the live GUI)."""

import cv2
import gym
import time
import math
import airsim
import keyboard
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from PyQt5 import QtCore
from collections import deque
from PyQt5.QtCore import pyqtSignal
from configparser import NoOptionError
from .multirotor_dynamics import MultirotorDynamicsAirsim
from .curriculum import (
    CurriculumManager,
    RollingOutcomeStats,
    categorize_outcome,
)

class AirsimGymEnv(gym.Env, QtCore.QThread):
    obs_signal      = pyqtSignal(int, np.ndarray)
    info_signal     = pyqtSignal(int, np.ndarray)
    action_signal   = pyqtSignal(int, np.ndarray)
    state_signal    = pyqtSignal(int, np.ndarray)
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)
    reward_signal   = pyqtSignal(int, float, float)
    pose_signal     = pyqtSignal(np.ndarray, np.ndarray)    
    lgmd_signal     = pyqtSignal(float, float, np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format},suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        self.model = None
        self.data_path = None
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.frame_history = 4
        self._obs_history = deque(maxlen=self.frame_history)
        self._pos_hist = deque(maxlen=3)      
        self._vel_hist = deque(maxlen=3)      
        self._yaw_hist = deque(maxlen=3)      
        self.action_smooth_enable = True  
        self.action_smooth_window = 4     
        self._action_history = None       
        self.uav_trajectory_list = []  
        self.goal_trajectory_list = []  
        self.info = None
        self.curriculum_manager = None
        self._episode_stats = {
            'success_count': 0,
            'total_count': 0
        }
        self.outcome_tracker = RollingOutcomeStats(window_size=20)

    def set_config(self, cfg):
        self.cfg = cfg
        self.env_name = cfg.get('options', 'env_name')
        self.dynamic_name = cfg.get('options', 'dynamic_name')
        self.perception_type = cfg.get('options', 'perception')
        try:
            self.dim = cfg.getint('options', 'dim')
        except:
            self.dim = 2
        self.lidar_points_per_layer = cfg.getint('environment', 'lidar_points_per_layer')
        self.lidar_num_layers = cfg.getint('environment', 'lidar_num_layers')
        self.lidar_max_range = cfg.getint('environment', 'lidar_max_range')
        self.VerticalFOVUpper = cfg.getint('environment', 'VerticalFOVUpper')
        self.VerticalFOVLower = cfg.getint('environment', 'VerticalFOVLower')
        self.HorizontalFOVStart = cfg.getint('environment', 'HorizontalFOVStart')
        self.HorizontalFOVEnd = cfg.getint('environment', 'HorizontalFOVEnd')
        self.acc_xy_max = cfg.getfloat('multirotor', 'acc_xy_max')
        self.acc_z_max = cfg.getfloat('multirotor', 'acc_z_max')
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_rad = math.radians(cfg.getfloat('multirotor', 'yaw_rate_max_deg'))
        self.goal_distance = cfg.getfloat('goal', 'goal_distance_init')
        self.max_episode_steps = cfg.getint('options', 'max_episode_steps')
        try:
            self.lidar_sensor_name = cfg.get('environment', 'lidar_sensor_name')
        except:
            self.lidar_sensor_name = "LidarSensor1"
        self.dynamic_model = MultirotorDynamicsAirsim(cfg)           
        self.dynamic_model.env = self
        if self.env_name == 'Easy_2D':
            start_position = [0, 0, 5]
            self.dynamic_model.set_start(start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(distance=self.goal_distance, random_angle=math.pi*2)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [-0.5, 10]
        elif self.env_name == 'NavRL_2D':
            start_position = [0, 0, 5]
            self.dynamic_model.set_start(start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(distance=self.goal_distance, random_angle=math.pi*2)
            self.work_space_x = [-50, 50]
            self.work_space_y = [-25, 25]
            self.work_space_z = [-0.5, 20]
        else:
            raise Exception("Invalid env_name!", self.env_name)
        self.client = self.dynamic_model.client
        cmd = "SpawnObstacles"
        self.client.simRunConsoleCommand(cmd)
        time.sleep(1)
        self.lidar_feature_length = cfg.getint('options', 'lidar_feature_length')
        self.uav_feature_length = cfg.getint('options', 'uav_feature_length')
        self.goal_feature_length = cfg.getint('options', 'goal_feature_length')
        self.single_frame_dim = (self.lidar_feature_length + 
                                 self.uav_feature_length + 
                                 self.goal_feature_length)
        self.w_p = cfg.getfloat('reward', 'w_p')
        self.w_t = cfg.getfloat('reward', 'w_t')
        self.w_s = cfg.getfloat('reward', 'w_s')
        self.w_e = cfg.getfloat('reward', 'w_e')
        self.w_j = cfg.getfloat('reward', 'w_j')
        self.w_yaw_align = cfg.getfloat('reward', 'w_yaw_align')
        self.w_yaw_rate = cfg.getfloat('reward', 'w_yaw_rate')
        self.w_yaw_trend = cfg.getfloat('reward', 'w_yaw_trend')
        self.w_vel_align = cfg.getfloat('reward', 'w_vel_align')
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0
        self.crash_distance = cfg.getfloat('environment', 'crash_distance')
        self.accept_radius = cfg.getfloat('environment', 'accept_radius')
        self.acc_max = cfg.getfloat('multirotor', 'acc_xy_max')
        self.d_crash = self.crash_distance
        self.d_max_lidar = self.lidar_max_range
        self.min_distance_to_obstacles = self.lidar_max_range
        if self.perception_type == 'lidar_vector':
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(1, self.single_frame_dim),
                dtype=np.float32)
        elif self.perception_type == 'lidar_vector_multi':
            total_len = self.frame_history * self.single_frame_dim
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(total_len,),
                dtype=np.float32)
        else:
            raise ValueError(f"Invalid perception_type: '{self.perception_type}'")
        self.action_space = self.dynamic_model.action_space
        self.action_dim = self.action_space.shape[0]
        self.action_smooth_enable = cfg.getboolean('options', 'action_smooth_enable')
        self.action_smooth_window = cfg.getint('options', 'action_smooth_window')
        self.use_safe_vel = cfg.getboolean('options', 'use_safe_vel')
        self.vel_safe_lidar_dist = cfg.getfloat('options', 'vel_safe_lidar_dist')
        self._action_history = [deque(maxlen=self.action_smooth_window)
                                for _ in range(self.action_dim)]
        self._action_smoothed_prev = np.zeros(self.action_dim, dtype=np.float32)
        self._action_smoothed_inited = np.zeros(self.action_dim, dtype=bool)
        self.reward_type = cfg.get('options', 'reward_type')
        self._dt = float(getattr(self.dynamic_model, 'dt'))
        try:
            curriculum_enabled = cfg.getboolean('curriculum', 'enabled')
            if curriculum_enabled:
                curriculum_config_path = cfg.get('curriculum', 'config_path', 
                                                fallback='config/curriculum_config.yaml')
                self.curriculum_manager = CurriculumManager(
                    config_path=curriculum_config_path,
                    airsim_client=self.client,
                    outcome_tracker=self.outcome_tracker
                )
            else:
                self.curriculum_manager = None
        except (NoOptionError, KeyError):
            self.curriculum_manager = None
        self.episode_trajectory_data = []

    def reset(self):
        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.uav_trajectory_list = []
        self.goal_trajectory_list = []
        self._pos_hist.clear()
        self._vel_hist.clear()
        self._yaw_hist.clear()
        self._obs_history.clear()
        if self._action_history is not None:
            for action_deque in self._action_history:
                action_deque.clear()
        self._action_smoothed_prev = np.zeros(self.action_dim, dtype=np.float32)
        self._action_smoothed_inited = np.zeros(self.action_dim, dtype=bool)
        if self.curriculum_manager and self.curriculum_manager.enabled:
            try:
                curriculum_info = self.curriculum_manager.reset()
                time.sleep(0.8)
            except Exception as e:
                pass
        try:
            self.client.enableApiControl(False)
            self.client.armDisarm(False)
        except:
            pass
        self.client.reset()
        self.dynamic_model.reset()
        time.sleep(0.5)
        try:
            _ = self.client.simGetCollisionInfo()
        except:
            pass
        self.dynamic_model.init_goal_pose_adaptive()
        yaw_noise = self.dynamic_model.start_random_angle * np.random.random()
        start_x = self.dynamic_model.start_position[0]
        start_y = self.dynamic_model.start_position[1]
        start_z = self.dynamic_model.start_position[2]
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = start_x
        pose.position.y_val = start_y
        pose.position.z_val = -start_z 
        pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
        self.client.simSetVehiclePose(pose, True)
        time.sleep(0.3)
        self.client.simPause(False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        try:
            self.client.moveToPositionAsync(
                x=start_x, y=start_y, z=-start_z-3.0, velocity=0.0, timeout_sec=5.0,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(yaw_noise))
            ).join()
            time.sleep(0.3)
            hover_start = time.time()
            while (time.time() - hover_start) < 1.0:
                self.client.moveByVelocityAsync(
                    vx=0, vy=0, vz=0, duration=0.1,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(yaw_noise))
                ).join()
                time.sleep(0.05)
        except Exception as e:
            pass
        self.client.simPause(True)
        try:
            p0 = np.array(self.dynamic_model.get_position(), dtype=np.float32)
            v0 = np.array(self.dynamic_model.get_velocity(), dtype=np.float32)
        except Exception as e:
            p0 = np.array([start_x, start_y, start_z], dtype=np.float32)
            v0 = np.zeros(3, dtype=np.float32)
        for _ in range(3):
            self._pos_hist.append(p0)
            self._vel_hist.append(v0)
            self._yaw_hist.append(0.0)
        try:
            single_obs = self.get_obs_lidar_vector().flatten()
            for _ in range(self.frame_history):
                self._obs_history.append(single_obs)
            _ = self.get_lidar_data(ver=1, hor=36)
        except Exception as e:
            self.min_distance_to_obstacles = self.lidar_max_range
        self.previous_distance_from_des_point = self.dynamic_model.get_current_distance_adaptive()
        initial_goal = list(self.dynamic_model.goal_position)
        self.goal_trajectory_list.append(initial_goal)
        obs = self.get_obs()
        self.episode_trajectory_data = []
        init_record = f"0,{self.dynamic_model.x:.4f},{self.dynamic_model.y:.4f},{self.dynamic_model.z:.4f}," \
                      f"{self.dynamic_model.vx:.4f},{self.dynamic_model.vy:.4f},{self.dynamic_model.vz:.4f}," \
                      f"{self.dynamic_model.goal_position[0]:.4f},{self.dynamic_model.goal_position[1]:.4f},{self.dynamic_model.goal_position[2]:.4f}\n"
        self.episode_trajectory_data.append(init_record)
        return obs

    def step(self, action):
        raw_action = action
        action = self._smooth_action_exponential(action)
        action = self._apply_safe_velocity_correction(action)
        self.dynamic_model.update_goal_pose_adaptive()
        self.dynamic_model.set_action(action)
        position_ue4 = self.dynamic_model.get_position()
        self.uav_trajectory_list.append(position_ue4)
        position_goal = list(self.dynamic_model.goal_position)
        self.goal_trajectory_list.append(position_goal)
        obs = self.get_obs()
        done, self.info = self.is_done()
        reward = self.compute_reward_math_model(done,self.info, action)
        self.info['step_num'] = self.step_num
        self.cumulated_episode_reward += reward
        if done and self.curriculum_manager is not None and self.curriculum_manager.enabled:
            level_changed, curriculum_info = self.curriculum_manager.update(
                done=done,
                episode_success=self.info.get('is_success', False),
                total_reward=self.cumulated_episode_reward,
                done_info=self.info
            )
            self.info.update({
                'curriculum_level_changed': level_changed,
                'current_level': curriculum_info.get('level', 0),
                'level_name': curriculum_info.get('name', 'Fixed'),
                'success_rate_20': curriculum_info.get('success_rate_20', 0.0),
                'crash_rate_20': curriculum_info.get('crash_rate_20', 0.0),
                'timeout_rate_20': curriculum_info.get('timeout_rate_20', 0.0),
                'workout_rate_20': curriculum_info.get('workout_rate_20', 0.0),
            })
            if level_changed:
                import os
                from datetime import datetime
                current_script_dir = os.path.dirname(os.path.abspath(__file__))
                log_dir = os.path.join(current_script_dir, "curriculum_level_logs")
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"Level_{curriculum_info['level']}_Record_{time_str}.txt"
                file_path = os.path.join(log_dir, filename)
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(f"System Time: {time_str}\n")
                    f.write(f"Curriculum Level: {curriculum_info['level']}\n")
                    f.write(f"Current Episode: {self.episode_num}\n")
                    f.write(f"Total Steps: {self.total_step}\n")
        elif done:
            outcome = categorize_outcome(self.info, self.info.get('is_success', False))
            self.outcome_tracker.add(outcome)
            self.info.update(self.outcome_tracker.rates())
            self.info['current_level'] = 0
            self.info['level_name'] = 'Fixed'
        try:
            p = np.array(self.dynamic_model.get_position(), dtype=np.float32)
            v = np.array(self.dynamic_model.get_velocity(), dtype=np.float32)
        except Exception:
            p = self._pos_hist[-1]; v = self._vel_hist[-1]
        self._pos_hist.append(p); self._vel_hist.append(v)
        yaw_rate_now = float(action[-1]) if len(action) > 0 else 0.0
        self._yaw_hist.append(yaw_rate_now)
        self.set_pyqt_signal_multirotor(action, reward)
        self.step_num += 1
        self.total_step += 1
        traj_record = f"{self.step_num},{self.dynamic_model.x:.4f},{self.dynamic_model.y:.4f},{self.dynamic_model.z:.4f}," \
                      f"{self.dynamic_model.vx:.4f},{self.dynamic_model.vy:.4f},{self.dynamic_model.vz:.4f}," \
                      f"{self.dynamic_model.goal_position[0]:.4f},{self.dynamic_model.goal_position[1]:.4f},{self.dynamic_model.goal_position[2]:.4f}\n"
        self.episode_trajectory_data.append(traj_record)
        if done:
            import os
            base_dir = getattr(self, "data_path", None)
            save_dir = os.path.join(base_dir, "trajectory_logs") if base_dir else "trajectory_logs"
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f"episode_{self.episode_num}.txt")
            with open(file_path, "w") as f:
                f.write("step,uav_x,uav_y,uav_z,uav_vx,uav_vy,uav_vz,goal_x,goal_y,goal_z\n")
                f.writelines(self.episode_trajectory_data)
        return obs, reward, done, self.info

    def get_obs(self):
        if self.perception_type == 'lidar_vector':
            obs = self.get_obs_lidar_vector()
        elif self.perception_type == 'lidar_vector_multi':
            obs = self.get_obs_lidar_vector_multi()
        else:
            obs = None
        return obs

    def get_lidar_data(self, ver: int = None, hor: int = None) -> np.ndarray:
        try:
            lidar_data = self.client.getLidarData(lidar_name=self.lidar_sensor_name)
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            if points.size == 0:
                if ver is None or hor is None \
                or (ver == self.lidar_num_layers and hor == self.lidar_points_per_layer):
                    shape = (self.lidar_num_layers, self.lidar_points_per_layer)
                else:
                    shape = (ver, hor)
                return np.ones(shape, dtype=np.float32) * self.lidar_max_range
            dists = np.linalg.norm(points, axis=1)
            horiz = np.arctan2(points[:,1], points[:,0])
            vert = np.arcsin(np.clip(points[:,2] / dists, -1, 1))
            h_range = np.radians(self.HorizontalFOVEnd - self.HorizontalFOVStart)
            v_range = np.radians(self.VerticalFOVUpper - self.VerticalFOVLower)
            h_off   = np.radians(self.HorizontalFOVStart)
            v_off   = np.radians(self.VerticalFOVLower)
            h_idx = np.floor(((horiz - h_off) % (2*np.pi)) / h_range 
                            * self.lidar_points_per_layer).astype(int)
            v_idx = np.floor((vert - v_off) / v_range 
                            * self.lidar_num_layers).astype(int)
            valid = (h_idx>=0)&(h_idx<self.lidar_points_per_layer) \
                    &(v_idx>=0)&(v_idx<self.lidar_num_layers)
            depth_full = np.ones((self.lidar_num_layers, self.lidar_points_per_layer),
                                dtype=np.float32) * self.lidar_max_range
            for i in np.where(valid)[0]:
                r, c = v_idx[i], h_idx[i]
                depth_full[r, c] = min(depth_full[r, c], dists[i])
            self.min_distance_to_obstacles = float(depth_full.min())
            if ver is None or hor is None \
            or (ver == self.lidar_num_layers and hor == self.lidar_points_per_layer):
                return depth_full
            sector_min = np.zeros((ver, hor), dtype=np.float32)
            rows = np.array_split(depth_full, ver, axis=0)
            for i, r_block in enumerate(rows):
                cols = np.array_split(r_block, hor, axis=1)
                for j, block in enumerate(cols):
                    sector_min[i, j] = block.min()
            self.min_distance_to_obstacles = float(sector_min.min())
            return sector_min
        except Exception as e:
            if ver is None or hor is None \
            or (ver == self.lidar_num_layers and hor == self.lidar_points_per_layer):
                shape = (self.lidar_num_layers, self.lidar_points_per_layer)
            else:
                shape = (ver, hor)
            return np.ones(shape, dtype=np.float32) * self.lidar_max_range

    def get_obs_lidar_vector(self):
        lidar_vec = self.get_lidar_data(ver=1, hor=self.lidar_feature_length).flatten()
        lidar_feature = 1.0 - np.clip(lidar_vec, 0, self.lidar_max_range) / self.lidar_max_range
        goal_feature, _, _ = self.dynamic_model.get_goal_feature_normalized()
        uav_feature = self.dynamic_model.get_uav_feature_normalized()
        single_obs = np.concatenate((lidar_feature, goal_feature, uav_feature), axis=0)
        self.dynamic_model.state_norm = np.concatenate((uav_feature, goal_feature))
        return single_obs.reshape(1, -1)

    def get_obs_lidar_vector_multi(self):
        lidar_vec = self.get_lidar_data(ver=1, hor=self.lidar_feature_length).flatten()
        lidar_feature = 1.0 - np.clip(lidar_vec, 0, self.lidar_max_range) / self.lidar_max_range
        goal_feature, _, _ = self.dynamic_model.get_goal_feature_normalized()
        uav_feature = self.dynamic_model.get_uav_feature_normalized()
        single_obs = np.concatenate((lidar_feature, goal_feature, uav_feature), axis=0)
        self._obs_history.append(single_obs)
        while len(self._obs_history) < self.frame_history:
            self._obs_history.appendleft(self._obs_history[0])
        stacked = np.concatenate(list(self._obs_history), axis=0)
        self.dynamic_model.state_norm = np.concatenate((uav_feature, goal_feature))
        return stacked.astype(np.float32)

    def compute_reward_math_model(self, done, done_info, action):
        if done:
            if done_info['is_success']: 
                return 200.0
            if done_info['is_crash'] or done_info['is_not_in_workspace']: 
                return -200.0 
            if done_info['timeout']:
                d_current = self.dynamic_model.get_current_distance_adaptive()
                d_init = max(self.dynamic_model.goal_distance_init, 1.0) 
                progress_ratio = 1.0 - np.clip(d_current / d_init, 0.0, 1.0)
                penalty_timeout = -50.0 - 150.0 * (1.0 - progress_ratio)
                return float(penalty_timeout)
            return -200.0 
        d_t = self.dynamic_model.get_current_distance_adaptive()
        d_tm1 = self.previous_distance_from_des_point
        self.previous_distance_from_des_point = d_t
        velocity = np.array(self.dynamic_model.get_velocity(), dtype=np.float32)
        speed = np.linalg.norm(velocity)
        psi_err = self.dynamic_model.current_psi_err 
        yaw_rate = self.dynamic_model.ang_vz 
        d_min = self.min_distance_to_obstacles
        pos_drone = np.array([self.dynamic_model.x, self.dynamic_model.y, -self.dynamic_model.z]) 
        pos_goal = np.array(self.dynamic_model.goal_position)
        vec_to_goal = pos_goal - pos_drone
        dist_to_goal = np.linalg.norm(vec_to_goal)
        r_progress = 0.0
        if dist_to_goal > 0.1:
            u_to_goal = vec_to_goal / dist_to_goal
            vel_projected = np.dot(velocity, u_to_goal)
            dist_gained = vel_projected * self._dt
            if self.dynamic_model.goal_distance_init > 0.1:
                r_progress = self.w_p * (dist_gained / self.dynamic_model.goal_distance_init)
            else:
                r_progress = 0.0
        r_time = self.w_t
        d_safe = self.d_crash + (speed**2) / (2 * self.acc_max)
        if d_min < d_safe:
            denom = max(d_safe, 0.01)
            exponent = min((d_safe - d_min) / denom, 10.0)
            r_safety = -self.w_s * (math.exp(exponent) - 1.0)
        else:
            r_safety = 0.0
        r_energy = 0.0
        r_jerk = 0.0
        if len(self._vel_hist) >= 1:
            v_t = velocity
            v_tm1 = self._vel_hist[-1]
            acc_vec_curr = (v_t - v_tm1) / self._dt
            r_energy = -self.w_e * np.dot(acc_vec_curr, acc_vec_curr)
            if len(self._vel_hist) >= 2:
                v_tm2 = self._vel_hist[-2]
                acc_vec_prev = (v_tm1 - v_tm2) / self._dt
                jerk_vec = (acc_vec_curr - acc_vec_prev) / self._dt
                r_jerk = -self.w_j * np.dot(jerk_vec, jerk_vec)
        r_yaw_align = self.w_yaw_align * np.exp(-2.0 * abs(psi_err))
        if np.sign(psi_err) == np.sign(yaw_rate):
            r_yaw_trend = self.w_yaw_trend * min(abs(yaw_rate), 1.0)
        else:
            r_yaw_trend = -self.w_yaw_trend * min(abs(yaw_rate), 1.0)
        r_yaw_damp = -self.w_yaw_rate * (yaw_rate ** 2)
        r_heading = r_yaw_align + r_yaw_trend + r_yaw_damp
        r_velocity_line = 0.0
        pos_curr = np.array([self.dynamic_model.x, self.dynamic_model.y, -self.dynamic_model.z])
        pos_goal = np.array(self.dynamic_model.goal_position) 
        if self.dim == 2:
            vec_goal = pos_goal[:2] - np.array([self.dynamic_model.x, self.dynamic_model.y])
            vec_vel = velocity[:2]
        else:
            vec_goal = pos_goal - np.array([self.dynamic_model.x, self.dynamic_model.y, self.dynamic_model.z])
            vec_vel = np.array([self.dynamic_model.vx, self.dynamic_model.vy, self.dynamic_model.vz])
        dist_to_goal = np.linalg.norm(vec_goal)
        if speed > 0.1 and dist_to_goal > 0.1:
            cos_sim = np.dot(vec_vel, vec_goal) / (speed * dist_to_goal)
            r_velocity_line = self.w_vel_align * cos_sim
        total_reward = (r_progress + r_time + r_safety + r_energy + r_jerk +r_heading + r_velocity_line)
        return float(total_reward)

    def is_done(self):
        episode_done = False
        info = {
            'is_success': False, 'is_crash': False, 'is_not_in_workspace': False, 
            'timeout': False, 'done_reason': None
        }
        if self.is_not_inside_workspace():
            episode_done = True; info['is_not_in_workspace'] = True; info['done_reason'] = 'out_of_workspace'
        elif self.is_in_desired_pose():
            episode_done = True; info['is_success'] = True; info['done_reason'] = 'success'
        elif self.is_crashed():
            episode_done = True; info['is_crash'] = True; info['done_reason'] = 'collision'
        elif self.step_num >= self.max_episode_steps:
            episode_done = True; info['timeout'] = True; info['done_reason'] = 'timeout'
        return episode_done, info

    def is_not_inside_workspace(self):
        cp = self.dynamic_model.get_position()
        return (cp[0] < self.work_space_x[0] or cp[0] > self.work_space_x[1] or 
                cp[1] < self.work_space_y[0] or cp[1] > self.work_space_y[1] or 
                cp[2] < self.work_space_z[0] or cp[2] > self.work_space_z[1])

    def is_in_desired_pose(self):
        return self.dynamic_model.get_current_distance_adaptive() < self.accept_radius

    def is_crashed(self):
        collision_info = self.client.simGetCollisionInfo()
        min_dist_condition = (self.min_distance_to_obstacles < self.crash_distance and 
                              self.min_distance_to_obstacles < self.lidar_max_range)
        return collision_info.has_collided or min_dist_condition

    def _smooth_action_exponential(self, raw_action):
        if not self.action_smooth_enable:
            return raw_action
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        out = np.empty_like(raw, dtype=np.float32)
        alpha = float(2.0 / (self.action_smooth_window + 1.0))
        for i in range(self.action_dim):
            self._action_history[i].append(raw[i])
            if not self._action_smoothed_inited[i]:
                s = raw[i] 
                self._action_smoothed_inited[i] = True
            else:
                s_prev = self._action_smoothed_prev[i]
                s = alpha * raw[i] + (1.0 - alpha) * s_prev
            if hasattr(self, 'action_space'):
                low_i = float(self.action_space.low[i])
                high_i = float(self.action_space.high[i])
                s = float(np.clip(s, low_i, high_i))
            out[i] = s
            self._action_smoothed_prev[i] = s 
        return out

    def _apply_safe_velocity_correction(self, action):
        if (not getattr(self, "use_safe_vel", False)) or getattr(self, "dim", 2) != 2:
            return action
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if act.shape[0] < 3:
            return action
        num_bins = int(getattr(self, "lidar_points_per_layer", 0))
        if num_bins <= 0:
            return action
        dt = float(getattr(self, "_dt", 0.1))
        dt = max(dt, 1e-3)
        v_xy_max = float(getattr(self, "v_xy_max", getattr(self.dynamic_model, "v_xy_max", 1.0)))
        v_xy_max = max(v_xy_max, 1e-3)
        a_max = float(getattr(self, "acc_max", getattr(self.dynamic_model, "acc_xy_max", 1.0)))
        a_max = max(a_max, 0.1)
        lidar_max_range = float(getattr(self, "lidar_max_range", 50.0))
        crash_distance = float(getattr(self, "crash_distance", 2.0))
        trigger_dist = float(getattr(self, "vel_safe_lidar_dist", 8.0))
        buffer_dist = crash_distance + 0.8
        trigger_dist = max(trigger_dist, buffer_dist + 0.2)
        def _quat_mul(q1, q2):
            w1, x1, y1, z1 = q1.w_val, q1.x_val, q1.y_val, q1.z_val
            w2, x2, y2, z2 = q2.w_val, q2.x_val, q2.y_val, q2.z_val
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return airsim.Quaternionr(x_val=x, y_val=y, z_val=z, w_val=w)
        def _quat_to_rotmat(q):
            w, x, y, z = float(q.w_val), float(q.x_val), float(q.y_val), float(q.z_val)
            n = w * w + x * x + y * y + z * z
            if n < 1e-12:
                return np.eye(3, dtype=np.float32)
            s = 2.0 / n
            wx, wy, wz = s * w * x, s * w * y, s * w * z
            xx, xy, xz = s * x * x, s * x * y, s * x * z
            yy, yz, zz = s * y * y, s * y * z, s * z * z
            return np.array(
                [
                    [1.0 - (yy + zz), xy - wz, xz + wy],
                    [xy + wz, 1.0 - (xx + zz), yz - wx],
                    [xz - wy, yz + wx, 1.0 - (xx + yy)],
                ],
                dtype=np.float32,
            )
        v_now_world = np.zeros(2, dtype=np.float32)
        q_wv = None
        vehicle_pos = None
        try:
            state = self.client.getMultirotorState()
            kin = state.kinematics_estimated
            v_now_world[:] = [kin.linear_velocity.x_val, kin.linear_velocity.y_val]
            q_wv = kin.orientation
            vehicle_pos = np.array([kin.position.x_val, kin.position.y_val, kin.position.z_val], dtype=np.float32)
        except Exception:
            pass
        try:
            lidar_data = self.client.getLidarData(lidar_name=self.lidar_sensor_name)
        except Exception:
            safe = act.copy()
            safe[:2] = 0.0
            safe[2] = 0.0
            return safe
        if lidar_data is None or len(lidar_data.point_cloud) < 3:
            safe = act.copy()
            safe[:2] = 0.0
            safe[2] = 0.0
            return safe
        lidar_pos = np.array(
            [lidar_data.pose.position.x_val, lidar_data.pose.position.y_val, lidar_data.pose.position.z_val],
            dtype=np.float32,
        )
        lidar_pose_is_world = False
        if vehicle_pos is not None:
            lidar_pose_is_world = float(np.linalg.norm(lidar_pos - vehicle_pos)) < 5.0
        q_l = lidar_data.pose.orientation
        if lidar_pose_is_world or (q_wv is None):
            q_wl = q_l
        else:
            q_wl = _quat_mul(q_wv, q_l)
        R_wl = _quat_to_rotmat(q_wl)
        R_lw = R_wl.T
        points = np.asarray(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        dists = np.linalg.norm(points, axis=1)
        valid = np.isfinite(dists) & (dists > 1e-3)
        if not np.any(valid):
            safe = act.copy()
            safe[:2] = 0.0
            safe[2] = 0.0
            return safe
        points = points[valid]
        dists = dists[valid]
        horiz = np.arctan2(points[:, 1], points[:, 0])
        h_off = math.radians(float(getattr(self, "HorizontalFOVStart", -180.0)))
        h_range = math.radians(float(getattr(self, "HorizontalFOVEnd", 180.0)) - float(getattr(self, "HorizontalFOVStart", -180.0)))
        if h_range <= 0.0:
            h_range = h_range % (2.0 * math.pi)
        if h_range <= 1e-6:
            h_range = 2.0 * math.pi
        full_fov = abs(h_range - 2.0 * math.pi) < 1e-3
        frac = ((horiz - h_off) % (2.0 * math.pi)) / h_range
        idx = np.floor(frac * num_bins).astype(np.int32)
        idx_valid = (idx >= 0) & (idx < num_bins)
        lidar_1d = np.full((num_bins,), lidar_max_range, dtype=np.float32)
        np.minimum.at(lidar_1d, idx[idx_valid], dists[idx_valid])
        lidar_1d = np.nan_to_num(lidar_1d, nan=0.0, posinf=lidar_max_range, neginf=0.0)
        lidar_1d = np.clip(lidar_1d, 0.0, lidar_max_range)
        d_global_min = float(np.min(lidar_1d))
        inflation = crash_distance + 0.5
        min_half_angle = math.radians(15.0)  
        half_angle = max(min_half_angle, math.atan2(inflation, max(trigger_dist, 0.1)))
        half_bins = int(math.ceil(half_angle / h_range * num_bins))
        half_bins = int(np.clip(half_bins, 1, max(1, num_bins // 2)))
        offsets = np.arange(-half_bins, half_bins + 1, dtype=np.int32)
        def corridor_min(center_idx: int) -> float:
            inds = (center_idx + offsets) % num_bins
            return float(np.min(lidar_1d[inds]))
        def max_safe_speed(clearance_m: float) -> float:
            available = clearance_m - buffer_dist
            if available <= 0.0:
                return 0.0
            disc = (a_max * dt) ** 2 + 2.0 * a_max * available
            if disc <= 0.0:
                return 0.0
            return max(0.0, -a_max * dt + math.sqrt(disc))
        v_cmd_norm = act[:2].astype(np.float32)
        if float(np.linalg.norm(v_cmd_norm)) >= 0.05:
            v_dir_world = v_cmd_norm
        else:
            v_dir_world = v_now_world
        if float(np.linalg.norm(v_dir_world)) < 1e-3:
            return action
        v_dir_lidar = R_lw.dot(np.array([v_dir_world[0], v_dir_world[1], 0.0], dtype=np.float32))
        dir_angle = float(math.atan2(v_dir_lidar[1], v_dir_lidar[0]))
        delta = (dir_angle - h_off) % (2.0 * math.pi)
        if (not full_fov) and (delta > h_range + 1e-6):
            safe = act.copy()
            safe[:2] = 0.0
            safe[2] = 0.0
            return safe
        desired_idx = int(np.clip(int((delta / h_range) * num_bins), 0, num_bins - 1))
        d_dir = corridor_min(desired_idx)
        v_des = float(np.linalg.norm(v_cmd_norm) * v_xy_max)
        v_des = min(v_des, v_xy_max * math.sqrt(2.0))
        v_safe_dir = max_safe_speed(d_dir) * 0.85  
        panic = d_global_min < (buffer_dist + 0.2)
        need_turn = panic or (d_dir < trigger_dist)
        if not need_turn:
            if v_des <= v_safe_dir + 1e-3:
                return action
            norm = float(np.linalg.norm(v_cmd_norm))
            if norm < 1e-6:
                safe = act.copy()
                safe[:2] = 0.0
                safe[2] = 0.0
                return safe
            unit = v_cmd_norm / norm
            v_cmd = max(0.0, min(v_des, v_safe_dir))
            safe = act.copy()
            safe[0] = float(np.clip(unit[0] * v_cmd / v_xy_max, -1.0, 1.0))
            safe[1] = float(np.clip(unit[1] * v_cmd / v_xy_max, -1.0, 1.0))
            return safe
        best_idx = None
        best_v = 0.0
        best_clear = 0.0
        for i in range(num_bins):
            clear_i = corridor_min(i)
            v_i = max_safe_speed(clear_i)
            if (v_i > best_v + 1e-6) or (abs(v_i - best_v) <= 1e-6 and clear_i > best_clear):
                best_v = v_i
                best_clear = clear_i
                best_idx = i
        if best_idx is None or best_v <= 1e-3:
            safe = act.copy()
            safe[:2] = 0.0
            safe[2] = 0.0
            return safe
        v_cmd = min(v_des, best_v * 0.85)
        if v_cmd < 0.05:
            safe = act.copy()
            safe[:2] = 0.0
            safe[2] = 0.0
            return safe
        angle_best = h_off + ((best_idx + 0.5) / num_bins) * h_range
        v_lidar_unit = np.array([math.cos(angle_best), math.sin(angle_best), 0.0], dtype=np.float32)
        v_world = R_wl.dot(v_lidar_unit)
        v_world_xy = v_world[:2].astype(np.float32)
        norm_xy = float(np.linalg.norm(v_world_xy))
        if norm_xy < 1e-6:
            safe = act.copy()
            safe[:2] = 0.0
            safe[2] = 0.0
            return safe
        unit_xy = v_world_xy / norm_xy
        safe = act.copy()
        safe[0] = float(np.clip(unit_xy[0] * v_cmd / v_xy_max, -1.0, 1.0))
        safe[1] = float(np.clip(unit_xy[1] * v_cmd / v_xy_max, -1.0, 1.0))
        safe[2] = 0.0  
        return safe

    def get_distance_to_goal_3d(self):
        cp = self.dynamic_model.get_position()
        gp = self.dynamic_model.goal_position
        return math.sqrt((cp[0]-gp[0])**2 + (cp[1]-gp[1])**2 + (cp[2]-gp[2])**2)

    def set_pyqt_signal_multirotor(self, action, reward):
        step = int(self.total_step)
        dist_xy = self.dynamic_model.get_distance_to_goal_2d()
        current_z = self.dynamic_model.get_position()[2]
        goal_z = self.dynamic_model.goal_position[2]
        dist_z = abs(goal_z - current_z)
        rel_yaw_deg = math.degrees(self.dynamic_model.current_psi_err)
        vx = self.dynamic_model.vx
        vy = self.dynamic_model.vy
        vz = self.dynamic_model.vz
        v_xy = math.sqrt(vx**2 + vy**2)
        yaw_rate_deg = math.degrees(action[-1]) if len(action) > 0 else 0.0
        state_packet = np.array([dist_xy, dist_z, rel_yaw_deg, v_xy, vz, yaw_rate_deg], dtype=np.float32)
        self.state_signal.emit(step, state_packet)
        r_rad, p_rad, y_rad = self.dynamic_model.get_attitude()
        att_deg = np.array([math.degrees(r_rad), math.degrees(p_rad), math.degrees(y_rad)], dtype=np.float32)
        try:
            cmd_att = self.dynamic_model.get_attitude_cmd()
            att_cmd_deg = np.degrees(cmd_att)
        except:
            att_cmd_deg = np.zeros(3)
        self.attitude_signal.emit(step, att_deg, att_cmd_deg)
        self.action_signal.emit(step, action)
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(np.asarray(self.uav_trajectory_list), np.asarray(self.goal_trajectory_list))
        lidar_raw = self.get_lidar_data(ver=1, hor=self.lidar_feature_length).flatten()
        goal_raw = np.array([dist_xy, rel_yaw_deg], dtype=np.float32) 
        uav_raw = np.concatenate(([vx, vy, vz], att_deg)) 
        obs_raw_vector = np.concatenate((lidar_raw, goal_raw, uav_raw))
        self.obs_signal.emit(step, obs_raw_vector)
