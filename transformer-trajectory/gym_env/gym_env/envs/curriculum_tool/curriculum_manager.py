# curriculum_manager.py

import yaml
import logging
from typing import Dict, List, Optional, Tuple
import time


class CurriculumManager:
    """课程学习管理器 - 根据训练进度动态调整障碍物难度"""
    
    def __init__(self, config_path: str, airsim_client):
        """
        初始化课程管理器
        
        Args:
            config_path: curriculum_config.yaml 路径
            airsim_client: AirSim 客户端实例
        """
        self.logger = logging.getLogger(__name__)
        self.airsim_client = airsim_client
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.enabled = self.config['curriculum_learning']['enabled']
        self.levels = self.config['curriculum_learning']['levels']
        
        # 当前等级
        self.current_level = 0
        self.current_level_idx = 0  # 兼容
        
        # ========== 新增：缓存当前等级的障碍物命令 ==========
        self.current_obstacle_command = ""
        
        # 统计数据
        self.episode_count = 0
        self.success_count = 0
        self.total_reward = 0.0
        self.recent_episodes = []
        self.window_size = 20
        
        self.logger.info(f"✅ CurriculumManager initialized: {len(self.levels)} levels")
        if self.enabled:
            self._prepare_level(0)  # ← 改为 prepare（只准备参数，不生成）
    
    def reset(self) -> Dict:
        """
        Episode 开始时调用
        - 每次都重新生成障碍物（实现随机性）
        - 使用当前等级的参数
        
        Returns:
            Dict: 当前课程状态
        """
        # ========== 关键：每次 reset 都重新生成障碍物 ==========
        if self.enabled:
            self.regenerate_obstacles()
        
        if not self.enabled:
            return {
                'level': 0, 
                'level_name': 'Fixed',
                'progress': 1.0
            }
        
        return {
            'level': self.current_level,
            'level_name': self.levels[self.current_level]['name'],
            'success_rate': self.success_count / max(self.episode_count, 1),
            'episode_count': self.episode_count,
            'total_levels': len(self.levels)
        }
    
    def regenerate_obstacles(self):
        """
        重新生成障碍物（使用当前等级的参数）
        - 每次 reset 调用
        - 实现随机训练
        """
        if not self.current_obstacle_command:
            self.logger.warning("⚠️  No obstacle command prepared")
            return
        
        try:
            # 清空旧障碍物
            self.airsim_client.simRunConsoleCommand("ClearAllObstacles")
            time.sleep(0.2)
            
            # 生成新障碍物（使用相同参数，但位置随机）
            self.airsim_client.simRunConsoleCommand(self.current_obstacle_command)
            time.sleep(0.5)
            
            if self.episode_count % 10 == 0:  # 每10轮打印一次
                self.logger.info(f"🔄 Regenerated obstacles for Level {self.current_level + 1}")
        
        except Exception as e:
            self.logger.error(f"❌ Failed to regenerate obstacles: {e}")
    
    def update(self, done: bool, episode_success: bool, total_reward: float) -> Tuple[bool, Dict]:
        """
        每个 step 结束时调用，更新统计并检查是否晋级
        
        Args:
            done: episode 是否结束
            episode_success: 是否成功
            total_reward: 总奖励
        
        Returns:
            Tuple[bool, Dict]: (是否晋级, 课程信息)
        """
        if not self.enabled:
            return False, {'level': 0, 'level_name': 'Fixed'}
        
        # 只在 episode 结束时更新
        if not done:
            return False, self.get_current_level_info()
        
        # 更新统计
        self.episode_count += 1
        if episode_success:
            self.success_count += 1
            self.recent_episodes.append(1)
        else:
            self.recent_episodes.append(0)
        
        self.total_reward += total_reward
        
        # 保持滑动窗口
        if len(self.recent_episodes) > self.window_size:
            self.recent_episodes.pop(0)
        
        # 检查是否晋级
        level_changed = False
        if self._should_level_up():
            level_changed = self._level_up()
        
        return level_changed, self.get_current_level_info()
    
    def _should_level_up(self) -> bool:
        """判断是否应该晋级"""
        if self.current_level >= len(self.levels) - 1:
            return False
        
        current_config = self.levels[self.current_level]
        min_episodes = current_config.get('min_episodes', 20)
        
        if self.episode_count < min_episodes:
            return False
        
        # 计算成功率
        if len(self.recent_episodes) < self.window_size:
            success_rate = self.success_count / self.episode_count
        else:
            success_rate = sum(self.recent_episodes) / len(self.recent_episodes)
        
        threshold = current_config.get('trigger_success_rate', 0.7)
        
        return success_rate >= threshold
    
    def _level_up(self) -> bool:
        """执行晋级操作"""
        old_level = self.current_level
        self.current_level += 1
        self.current_level_idx = self.current_level
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎉 LEVEL UP! {old_level + 1} → {self.current_level + 1}")
        self.logger.info(f"📊 Statistics:")
        self.logger.info(f"   - Total episodes: {self.episode_count}")
        self.logger.info(f"   - Success rate: {self.success_count / self.episode_count:.2%}")
        self.logger.info(f"   - Avg reward: {self.total_reward / self.episode_count:.2f}")
        self.logger.info(f"{'='*60}\n")
        
        # ========== 关键：只准备新等级参数，不立即生成 ==========
        success = self._prepare_level(self.current_level)
        
        # 重置统计
        self.episode_count = 0
        self.success_count = 0
        self.total_reward = 0.0
        self.recent_episodes.clear()
        
        return success
    
    def _prepare_level(self, level_idx: int) -> bool:
        """
        准备指定等级的障碍物参数（不生成障碍物）
        - 等级切换时调用
        - 只保存命令，下次 reset 时才生成
        
        Args:
            level_idx: 等级索引
        
        Returns:
            bool: 是否准备成功
        """
        if level_idx >= len(self.levels):
            self.logger.warning(f"Invalid level index: {level_idx}")
            return False
        
        level = self.levels[level_idx]
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📚 Preparing Level {level_idx + 1}: {level['name']}")
        self.logger.info(f"{'='*60}")
        
        # 构建命令参数
        obstacle_configs = []
        
        for obs_config in level['obstacles']:
            mesh_name = obs_config['mesh_name']
            count = obs_config['count']
            
            # 支持独立 XYZ 缩放
            if 'scale_x' in obs_config and 'scale_y' in obs_config and 'scale_z' in obs_config:
                scale_x_min, scale_x_max = obs_config['scale_x']
                scale_y_min, scale_y_max = obs_config['scale_y']
                scale_z_min, scale_z_max = obs_config['scale_z']
            elif 'scale_x' in obs_config and 'scale_y' in obs_config:
                scale_x_min, scale_x_max = obs_config['scale_x']
                scale_y_min, scale_y_max = obs_config['scale_y']
                scale_z_min, scale_z_max = obs_config.get('scale_range', [1.0, 1.0])
            else:
                scale_range = obs_config.get('scale_range', [1.0, 1.0])
                scale_x_min = scale_x_max = scale_range[0]
                scale_y_min = scale_y_max = scale_range[0]
                scale_z_min = scale_z_max = scale_range[1]
            
            config_str = (f"{mesh_name}:{count}:"
                         f"{scale_x_min:.2f}:{scale_x_max:.2f}:"
                         f"{scale_y_min:.2f}:{scale_y_max:.2f}:"
                         f"{scale_z_min:.2f}:{scale_z_max:.2f}")
            obstacle_configs.append(config_str)
            
            self.logger.info(f"  🌲 {mesh_name}: {count}x "
                            f"(X: {scale_x_min:.2f}-{scale_x_max:.2f}, "
                            f"Y: {scale_y_min:.2f}-{scale_y_max:.2f}, "
                            f"Z: {scale_z_min:.2f}-{scale_z_max:.2f})")
        
        # ========== 关键：只保存命令，不发送 ==========
        self.current_obstacle_command = "SpawnObstaclesFromJSON " + ",".join(obstacle_configs)
        self.logger.info(f"✅ Level parameters prepared (will generate on next reset)")
        
        return True
    
    def get_current_level_info(self) -> Dict:
        """获取当前课程等级信息"""
        if not self.enabled:
            return {
                'name': 'Fixed',
                'level': 0,
                'total_levels': 1,
                'success_rate': 0.0,
                'episode_count': 0
            }
        
        return {
            'name': self.levels[self.current_level]['name'],
            'level': self.current_level,
            'total_levels': len(self.levels),
            'success_rate': self.success_count / max(self.episode_count, 1),
            'episode_count': self.episode_count,
            'avg_reward': self.total_reward / max(self.episode_count, 1)
        }
    
    def force_level(self, level_idx: int):
        """强制切换到指定等级（调试用）"""
        if 0 <= level_idx < len(self.levels):
            self.current_level = level_idx
            self.current_level_idx = level_idx
            self.episode_count = 0
            self.success_count = 0
            self.total_reward = 0.0
            self.recent_episodes.clear()
            self._prepare_level(level_idx)
        else:
            self.logger.error(f"Invalid level index: {level_idx}")
