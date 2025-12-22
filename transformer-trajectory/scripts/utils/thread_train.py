"""
DRL训练主程序 - 支持 PPO/SAC/TD3
"""
import os
import sys
import ast
import random
import datetime
import argparse
from configparser import ConfigParser

import gym
import wandb
import numpy as np
import torch as th
from PyQt5 import QtCore

import gym_env
from .custom_policy_sb3 import MLP_encoder, MLP_embedding_Transformer_encoder

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.ppo.policies import MlpPolicy as MlpPolicy_ppo
from stable_baselines3.sac.policies import MlpPolicy as MlpPolicy_sac
from stable_baselines3.td3.policies import MlpPolicy as MlpPolicy_td3
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


def get_parser():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description="DRL训练程序 (支持 PPO/SAC/TD3)")
    parser.add_argument('-c', '--config', help='配置文件名(不含.ini后缀)',
                        default='config_Trees_SimpleMultirotor_2D')
    parser.add_argument('-n', '--note', help='训练备注信息', default='default_training')
    return parser


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                path = os.path.join(self.save_path, f"vec_normalize_{self.num_timesteps}_steps.pkl")
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saved VecNormalize stats to {path}")
        return True


class EpisodeStatsCallback(BaseCallback):
    """
    记录最近20场的终止比例（success/crash/timeout/workout）以及课程等级。
    利用 env info 中的统计，确保与课程管理一致。
    """
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True

        for done, info in zip(dones, infos):
            if not done or info is None:
                continue
            for key in ("success_rate_20", "crash_rate_20", "timeout_rate_20", "workout_rate_20"):
                if key in info:
                    self.logger.record(f"train/{key}", float(info[key]))
            if "current_level" in info:
                self.logger.record("curriculum/level", int(info["current_level"]))
            if "level_name" in info:
                self.logger.record("curriculum/level_name", info["level_name"])
        return True


class TrainingThread(QtCore.QThread):
    """DRL训练线程"""
    
    def __init__(self, config):
        super(TrainingThread, self).__init__()
        print("=" * 60)
        print("初始化训练线程")
        print("=" * 60)
        
        self.cfg = ConfigParser()
        self.cfg.read(config)
        self.project_name = self.cfg.get('options', 'env_name')
        print(f"项目: {self.project_name}")
        
        self.r_env = gym.make('airsim-env-v0')
        self.r_env.set_config(self.cfg)
        print(f"环境: {self.cfg.get('options', 'env_name')}")
        print(f"感知: {self.cfg.get('options', 'perception')}")

        self.monitor_env = Monitor(self.r_env)
        self.env = DummyVecEnv([lambda: self.monitor_env])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

        if self.cfg.getboolean('options', 'use_wandb'):
            wandb.init(
                project=self.project_name,
                notes=self.cfg.get('wandb', 'notes'),
                name=self.cfg.get('wandb', 'name'),
                sync_tensorboard=True,
                save_code=True
            )
            print("✓ WandB已启用")
        else:
            print("✓ WandB未启用")
    
    def terminate(self):
        print('训练线程已终止')
    
    def run(self):
        """主训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        seed = self._set_random_seed()
        log_path, model_path, config_path, data_path = self._create_directories(seed)
        
        with open(os.path.join(config_path, 'config.ini'), 'w') as f:
            self.cfg.write(f)
        print(f"✓ 配置已保存")
        
        policy_kwargs, net_arch_list = self._build_policy_kwargs()
        
        algo = self.cfg.get('options', 'algo')
        model = self._create_model(algo, policy_kwargs, net_arch_list, log_path, seed)
        
        self._train_model(model, model_path, data_path)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
    
    def _set_random_seed(self):
        seed = random.randint(0, 2**31 - 1)
        print(f"\n随机种子: {seed}")
        
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        
        if th.cuda.is_available():
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
            print("✓ CUDA种子已设置")
        
        return seed
    
    def _create_directories(self, seed):
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        base_path = (
            f"logs/{self.project_name}/"
            f"{timestamp}_seed{seed}_"
            f"{self.cfg.get('options', 'dynamic_name')}_"
            f"{self.cfg.get('options', 'policy_name')}_"
            f"{self.cfg.get('options', 'algo')}"
        )
        
        paths = {
            'log': os.path.join(base_path, 'tb_logs'),
            'model': os.path.join(base_path, 'models'),
            'config': os.path.join(base_path, 'config'),
            'data': os.path.join(base_path, 'data')
        }
        
        for name, path in paths.items():
            os.makedirs(path, exist_ok=True)
        
        print(f"✓ 目录已创建: {base_path}")
        return paths['log'], paths['model'], paths['config'], paths['data']
    
    def _build_policy_kwargs(self):
        print("\n" + "-" * 60)
        print("配置策略网络")
        print("-" * 60)
        
        l_dim = self.cfg.getint('options', 'lidar_feature_length') 
        u_dim = self.cfg.getint('options', 'uav_feature_length')   
        g_dim = self.cfg.getint('options', 'goal_feature_length')  
        single_frame_dim = l_dim + u_dim + g_dim
        frame_history = self.cfg.getint('options', 'frame_history')
        total_dim = single_frame_dim * frame_history
        
        print(f"特征结构(单帧): Lidar={l_dim}, UAV={u_dim}, Goal={g_dim}")
        print(f"帧历史: {frame_history}, 总输入维度: {total_dim}")
                
        activation_fn = th.nn.Tanh if self.cfg.get('options', 'activation_function') == 'tanh' else th.nn.ReLU
        
        policy_name = self.cfg.get('options', 'policy_name')
        policy_map = {
            'MLP_encoder': MLP_encoder,
            'MLP_embedding_Transformer_encoder': MLP_embedding_Transformer_encoder
        }
        if policy_name not in policy_map:
            raise ValueError(f"未知策略类型: {policy_name}")
        features_extractor_class = policy_map[policy_name]
        
        net_arch_list = ast.literal_eval(self.cfg.get("options", "net_arch"))
        print(f"网络架构: {net_arch_list}")
        
        policy_kwargs = {
            'features_extractor_class': features_extractor_class,
            'features_extractor_kwargs': {
                'features_dim': total_dim,
                'lidar_feature_dim': l_dim,
                'uav_feature_dim': u_dim,
                'goal_feature_dim': g_dim,
                'frame_history': frame_history
            },
            'activation_fn': activation_fn
        }
        
        return policy_kwargs, net_arch_list
    
    def _create_model(self, algo, policy_kwargs_base, net_arch_list, log_path, seed):
        print("\n" + "-" * 60)
        print(f"创建模型: {algo}")
        print("-" * 60)
        
        model_creators = {
            'PPO': self._create_ppo_model,
            'SAC': self._create_sac_model,
            'TD3': self._create_td3_model
        }
        
        if algo not in model_creators:
            raise ValueError(f"不支持的算法: {algo}")
        
        model = model_creators[algo](policy_kwargs_base, net_arch_list, log_path, seed)
        self._print_model_structure(model, algo)
        
        return model
    
    def _create_ppo_model(self, policy_kwargs_base, net_arch_list, log_path, seed):
        policy_kwargs = policy_kwargs_base.copy()
        policy_kwargs['net_arch'] = [dict(pi=net_arch_list, vf=net_arch_list)]
        return PPO(MlpPolicy_ppo, self.env, n_steps=self.cfg.getint('PPO', 'n_steps'), n_epochs=self.cfg.getint('PPO', 'n_epochs'), batch_size=self.cfg.getint('PPO', 'batch_size'), learning_rate=self.cfg.getfloat('PPO', 'learning_rate'), gamma=self.cfg.getfloat('PPO', 'gamma'), gae_lambda=self.cfg.getfloat('PPO', 'gae_lambda'), clip_range=self.cfg.getfloat('PPO', 'clip_range'), ent_coef=self.cfg.getfloat('PPO', 'ent_coef'), vf_coef=self.cfg.getfloat('PPO', 'vf_coef'), max_grad_norm=self.cfg.getfloat('PPO', 'max_grad_norm'), policy_kwargs=policy_kwargs, tensorboard_log=log_path, seed=seed, verbose=2)

    def _create_sac_model(self, policy_kwargs_base, net_arch_list, log_path, seed):
        policy_kwargs = policy_kwargs_base.copy()
        policy_kwargs['share_features_extractor'] = False
        policy_kwargs['net_arch'] = dict(pi=net_arch_list, qf=net_arch_list)
        return SAC(MlpPolicy_sac, self.env, buffer_size=self.cfg.getint('SAC', 'buffer_size'), learning_starts=self.cfg.getint('SAC', 'learning_starts'), batch_size=self.cfg.getint('SAC', 'batch_size'), learning_rate=self.cfg.getfloat('SAC', 'learning_rate'), gamma=self.cfg.getfloat('SAC', 'gamma'), train_freq=(self.cfg.getint('SAC', 'train_freq'), 'step'), gradient_steps=self.cfg.getint('SAC', 'gradient_steps'), tau=self.cfg.getfloat('SAC', 'tau'), ent_coef=self.cfg.get('SAC', 'ent_coef'), target_entropy=self.cfg.get('SAC', 'target_entropy'), policy_kwargs=policy_kwargs, tensorboard_log=log_path, seed=seed, verbose=2)

    def _create_td3_model(self, policy_kwargs_base, net_arch_list, log_path, seed):
        policy_kwargs = policy_kwargs_base.copy()
        policy_kwargs['share_features_extractor'] = False
        policy_kwargs['net_arch'] = dict(pi=net_arch_list, qf=net_arch_list)
        n_actions = self.env.action_space.shape[-1]
        noise_sigma = self.cfg.getfloat('TD3', 'action_noise_sigma') * np.ones(n_actions)
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma)
        return TD3(MlpPolicy_td3, self.env, action_noise=action_noise, buffer_size=self.cfg.getint('TD3', 'buffer_size'), learning_starts=self.cfg.getint('TD3', 'learning_starts'), batch_size=self.cfg.getint('TD3', 'batch_size'), learning_rate=self.cfg.getfloat('TD3', 'learning_rate'), gamma=self.cfg.getfloat('TD3', 'gamma'), train_freq=(self.cfg.getint('TD3', 'train_freq'), 'step'), gradient_steps=self.cfg.getint('TD3', 'gradient_steps'), tau=self.cfg.getfloat('TD3', 'tau'), policy_delay=self.cfg.getint('TD3', 'policy_delay'), target_policy_noise=self.cfg.getfloat('TD3', 'target_policy_noise'), target_noise_clip=self.cfg.getfloat('TD3', 'target_noise_clip'), policy_kwargs=policy_kwargs, tensorboard_log=log_path, seed=seed, verbose=2)

    def _print_model_structure(self, model, algo):
        print("\n" + "=" * 60)
        print("模型结构")
        print("=" * 60)
        print(f"算法: {algo}")
        print(f"Policy类: {model.policy.__class__.__name__}")
        print(f"特征提取器: {model.policy.features_extractor.__class__.__name__}")
        if algo == 'PPO':
            print("\n--- PPO架构 ---")
            print(f"MLP Extractor: {model.policy.mlp_extractor}")
            print(f"Action Net: {model.policy.action_net}")
            print(f"Value Net: {model.policy.value_net}")
        else:
            print("\n--- Off-Policy架构 ---")
            print(f"Actor: {model.policy.actor}")
            print(f"Critic: {model.policy.critic}")
        print("\n完整Policy:")
        print(model.policy)
        print("=" * 60)
    
    def _train_model(self, model, model_path, data_path):
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        self.r_env.model = model 
        self.r_env.data_path = data_path
        
        callbacks = self._setup_callbacks(model_path)
        
        total_timesteps = self.cfg.getint('options', 'total_timesteps')
        print(f"总训练步数: {total_timesteps:,}")
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=1,
            callback=CallbackList(callbacks)
        )
        
        final_model_path = os.path.join(model_path, f"model_sb3_final_{model.num_timesteps}_steps")
        model.save(final_model_path)
        final_stats_path = os.path.join(model_path, f"vec_normalize_final_{model.num_timesteps}_steps.pkl")
        self.env.save(final_stats_path)
        print(f"\n✓ 最终模型及统计数据已保存: {final_model_path}")
    
    def _setup_callbacks(self, model_path):
        callbacks = []
        save_freq = self.cfg.getint('options', 'save_freq')

        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=model_path,
            name_prefix='model_sb3'
        )
        callbacks.append(checkpoint_cb)

        vec_norm_cb = SaveVecNormalizeCallback(
            save_freq=save_freq,
            save_path=model_path
        )
        callbacks.append(vec_norm_cb)

        # 新增：课程/终止率 + Level 的 TB 记录
        callbacks.append(EpisodeStatsCallback())
        
        print(f"✓ 检查点频率: 每{save_freq}步")
        
        if self.cfg.getboolean('options', 'use_wandb'):
            wandb_cb = WandbCallback(model_save_freq=0, verbose=2)
            callbacks.append(wandb_cb)
            print("✓ WandB回调已配置")
        
        return callbacks


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    config_file = f'configs/{args.config}.ini'
    print(f"配置文件: {config_file}")
    
    training_thread = TrainingThread(config_file)
    training_thread.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n系统退出')
