"""Evaluation worker thread (legacy): runs evaluation episodes with the live GUI."""

from PyQt5 import QtCore
from configparser import ConfigParser
from stable_baselines3 import TD3, SAC, PPO
# 【新增】引入 VecEnv 组件 和 Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np
import gym_env
import gym
import math
import os
import sys
import time
from tqdm import tqdm
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


def rule_based_policy(obs):
    # 保持原样，省略内容...
    '''
    custom linear policy
    used for LGMD compare
    '''
    action = 0
    # 将obs从1~-1转换成0~1
    obs = np.squeeze(obs, axis=0)

    for i in range(5):
        obs[i] = obs[i]/2 + 0.5

    # obs_weight_depth = np.array([1.0, 3.0, 5.0, -3.0, -1.0, 3.0])
    obs_weight = np.array([1.0, 3.0, 3.0, -3.0, -1.0, 3.0])
    action = obs * obs_weight

    action_sum = np.sum(action)

    if action_sum > math.radians(40):
        action_sum = math.radians(40)
    elif action_sum < -math.radians(40):
        action_sum = -math.radians(40)

    return np.array([action_sum])


class EvaluateThread(QtCore.QThread):
    # signals
    def __init__(self, eval_path, config, model_file, eval_ep_num, eval_env=None, eval_dynamics=None):
        super(EvaluateThread, self).__init__()
        print("init evaluation thread")

        # config
        self.cfg = ConfigParser()
        self.cfg.read(config)

        if eval_env == 'NH_center':
            self.cfg.set('environment', 'accept_radius', str(1))

        self.eval_path = eval_path
        self.model_file = model_file
        self.eval_ep_num = eval_ep_num
        self.eval_env = self.cfg.get('options', 'env_name')
        self.eval_dynamics = self.cfg.get('options', 'dynamic_name')

        # 【修改】环境初始化
        # 1. 创建原始环境 (保留引用给GUI)
        self.r_env = gym.make('airsim-env-v0')
        self.r_env.set_config(self.cfg)
        
        # 2. 【新增】使用 Monitor 包裹，保持和训练时一致的结构
        self.monitor_env = Monitor(self.r_env)

        # 3. 包装成 DummyVecEnv
        self.env = DummyVecEnv([lambda: self.monitor_env])

        # 4. 【关键】加载 VecNormalize 统计数据
        # 假设统计文件名为 vec_normalize_xxxx_steps.pkl
        # 我们需要根据 model_file 的名字推断 stats file 的名字
        
        model_dir = os.path.dirname(self.model_file)
        model_name = os.path.basename(self.model_file)
        
        # 简单的字符串替换逻辑，尝试找到对应的 pkl 文件
        # 这里假设保存格式为: model_sb3_X_steps.zip -> vec_normalize_X_steps.pkl
        stats_name = model_name.replace("model_sb3", "vec_normalize").replace(".zip", ".pkl")
        stats_path = os.path.join(model_dir, stats_name)

        if os.path.exists(stats_path):
            print(f"Loading VecNormalize stats from: {stats_path}")
            self.env = VecNormalize.load(stats_path, self.env)
            # 【绝对关键】停止更新统计数据，停止归一化奖励（为了看真实得分）
            self.env.training = False 
            self.env.norm_reward = False
        else:
            print("Warning: No VecNormalize stats found! Evaluation might fail if model was trained with it.")
            # 如果没找到特定的步数文件，尝试找 final
            if "final" not in model_name:
                 print("Trying to find manual matches...")
                 # 这里可以根据你的实际情况添加容错逻辑

    def terminate(self):
        print('Evaluation terminated')

    def run(self):
        # self.run_rule_policy()
        return self.run_drl_model()

    def run_drl_model(self):
        print('start evaluation')
        algo = self.cfg.get('options', 'algo')
        
        # 加载模型时传入 env (DummyVecEnv 包装后的)
        if algo == 'TD3':
            model = TD3.load(self.model_file, env=self.env)
        elif algo == 'SAC':
            model = SAC.load(self.model_file, env=self.env)
        elif algo == 'PPO':
            model = PPO.load(self.model_file, env=self.env)
        else:
            raise Exception('algo set error {}'.format(algo))
        
        # model 注入到 raw environment (如果需要)
        self.r_env.model = model

        # VecEnv reset 返回的是 array，但 DummyVecEnv[0] 只有一个
        obs = self.env.reset() 
        
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        episode_successes = []
        episode_crashes = []
        traj_list_all = []
        action_list_all = []
        state_list_all = []
        obs_list_all = []

        traj_list = []
        action_list = []
        state_raw_list = []
        step_num_list = []
        obs_list = []

        while episode_num < self.eval_ep_num:
            # VecEnv 的 predict 不需要手动 wrap obs，但 obs 必须是 (n_envs, dim)
            unscaled_action, _ = model.predict(obs, deterministic=True)
            
            time_step += 1

            # 【修改】VecEnv step 返回的是 arrays
            # new_obs: (1, dim), reward: (1,), done: (1,), info: [{...}]
            new_obs, reward, done, infos = self.env.step(unscaled_action)
            
            # 提取单个值用于记录
            real_reward = reward[0]
            real_done = done[0]
            real_info = infos[0] # info 是 list of dicts
            real_action = unscaled_action[0] # 假设 action 是 (1, action_dim)

            # 获取数据用于绘图 (从 raw_env 获取)
            pose = self.r_env.dynamic_model.get_position()
            traj_list.append(pose)
            action_list.append(real_action)
            state_raw_list.append(self.r_env.dynamic_model.state_raw)
            # obs 是 Tensor 或 Array，保存这帧的输入
            obs_list.append(obs[0]) 

            obs = new_obs
            reward_sum[-1] += real_reward

            if real_done:
                episode_num += 1
                maybe_is_success = real_info.get('is_success')
                maybe_is_crash = real_info.get('is_crash')
                print('episode: ', episode_num, ' reward:', reward_sum[-1],
                      'success:', maybe_is_success)
                episode_successes.append(float(maybe_is_success))
                episode_crashes.append(float(maybe_is_crash))
                reward_sum = np.append(reward_sum, .0)
                
                # Reset 已经被 VecEnv 自动调用了吗？
                # SB3 的 VecEnv 在 done=True 时会自动 reset，返回新的 obs。
                # 但我们需要重置一些本地的列表
                
                # VecEnv 自动 reset，所以 obs 已经是新 episode 的第一帧了
                # 但是为了逻辑清晰，通常我们不需要手动 call reset，除非我们要强制重置
                # 这里保持 VecEnv 的自动 reset 机制
                
                if real_info.get('is_success'):
                    traj_list.append(1)
                    action_list.append(1)
                    step_num_list.append(real_info.get('step_num'))
                elif real_info.get('is_crash'):
                    traj_list.append(2)
                    action_list.append(2)
                else:
                    traj_list.append(3)
                    action_list.append(3)
                
                traj_list_all.append(traj_list)
                action_list_all.append(action_list)
                state_list_all.append(state_raw_list)
                obs_list_all.append(obs_list)
                
                traj_list = []
                action_list = []
                state_raw_list = []
                obs_list = []

        # save trajectory data ... (保持原样)
        eval_folder = self.eval_path + '/eval_{}_{}_{}'.format(self.eval_ep_num, self.eval_env, self.eval_dynamics)
        os.makedirs(eval_folder, exist_ok=True)
        np.save(eval_folder + '/traj_eval', np.array(traj_list_all, dtype=object))
        np.save(eval_folder + '/action_eval', np.array(action_list_all, dtype=object))
        np.save(eval_folder + '/state_eval', np.array(state_list_all, dtype=object))
        np.save(eval_folder + '/obs_eval', np.array(obs_list_all, dtype=object))

        print('Average episode reward: ', reward_sum[:self.eval_ep_num].mean(),
              'Success rate:', np.mean(episode_successes),
              'Crash rate: ', np.mean(episode_crashes),
              'average success step num: ', np.mean(step_num_list))
        
        results = [reward_sum[:self.eval_ep_num].mean(), np.mean(episode_successes), np.mean(episode_crashes), np.mean(step_num_list)]
        
        print(results)
        np.save(eval_folder + '/results', np.array(results))
        
        return results

    def run_rule_policy(self):
        obs = self.env.reset()
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        while episode_num < self.eval_ep_num:
            unscaled_action = rule_based_policy(obs)
            time_step += 1
            new_obs, reward, done, info, = self.env.step(unscaled_action)
            reward_sum[-1] += reward

            obs = new_obs
            if done:
                episode_num += 1
                maybe_is_success = info.get('is_success')
                print('episode: ', episode_num, ' reward:', reward_sum[-1],
                      'success:', maybe_is_success)
                reward_sum = np.append(reward_sum, .0)
                obs = self.env.reset()


def run_eval_multi():
    # run evaluation for multi models
    eval_logs_name = 'Maze'
    eval_logs_path = 'logs_eval/' + eval_logs_name
    eval_ep_num = 50
    eval_env_name = 'NH_center'        # 1-Trees 2-SimpleAvoid 3-NH_center
    eval_dynamic_name = 'SimpleMultirotor'  # 1-SimpleMultirotor or Multirotor

    model_list = []
    for train_name in os.listdir(eval_logs_path):
        for repeat_name in os.listdir(eval_logs_path + '/' + train_name):
            model_path = eval_logs_path + '/' + train_name + '/' + repeat_name
            model_list.append(model_path)
            # print(model_path)

    # evaluate model according to model path
    eval_num = len(model_list)
    results_list = []

    for i in tqdm(range(eval_num)):
        eval_path = model_list[i]
        config_file = eval_path + '/config.ini'
        model_file = eval_path + '/models/model_sb3.zip'

        print(i, eval_path)
        evaluate_thread = EvaluateThread(eval_path, config_file, model_file, eval_ep_num, eval_env_name, eval_dynamic_name)
        results = evaluate_thread.run()
        results_list.append(results)

        del evaluate_thread

    # save all results in a numpy file
    print(results_list)
    np.save('logs_eval/results/eval_{}_{}_{}_{}'.format(eval_ep_num, eval_logs_name, eval_env_name, eval_dynamic_name), np.array(results_list))


if __name__ == "__main__":
    try:
        # main()
        # 请确保 run_eval_multi() 调用时传递正确的参数
        from utils.thread_evaluation import run_eval_multi # 仅作示例
        run_eval_multi()
    except KeyboardInterrupt:
        print('system exit')
