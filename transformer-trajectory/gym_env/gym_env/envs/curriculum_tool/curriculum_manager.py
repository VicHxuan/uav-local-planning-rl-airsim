"""Curriculum manager (legacy): outcome statistics and difficulty scheduling."""

import yaml
import logging
from typing import Dict, List, Optional, Tuple
import time
from collections import Counter, deque


def categorize_outcome(done_info: Optional[Dict], episode_success: bool) -> str:
    """
    统一的终止类型判定:
    - success
    - crash
    - timeout
    - workout (out of workspace)
    - other
    """
    if episode_success:
        return "success"
    if done_info:
        if done_info.get("is_crash", False):
            return "crash"
        if done_info.get("timeout", False):
            return "timeout"
        if done_info.get("is_not_in_workspace", False):
            return "workout"
    return "other"


class RollingOutcomeStats:
    """维护最近 N 个 episode 的终止类型，并计算频率"""
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.outcomes = deque(maxlen=window_size)

    def add(self, outcome: str):
        self.outcomes.append(outcome)

    def reset(self):
        self.outcomes.clear()

    def rates(self) -> Dict[str, float]:
        total = len(self.outcomes)
        if total == 0:
            return {
                "success_rate_20": 0.0,
                "crash_rate_20": 0.0,
                "timeout_rate_20": 0.0,
                "workout_rate_20": 0.0,
            }
        counts = Counter(self.outcomes)
        denom = float(total)
        return {
            "success_rate_20": counts.get("success", 0) / denom,
            "crash_rate_20": counts.get("crash", 0) / denom,
            "timeout_rate_20": counts.get("timeout", 0) / denom,
            "workout_rate_20": counts.get("workout", 0) / denom,
        }


class CurriculumManager:
    """课程学习管理器 - 根据训练进度动态调整障碍物难度"""

    def __init__(self, config_path: str, airsim_client, outcome_tracker: Optional[RollingOutcomeStats] = None):
        """
        初始化课程管理器

        Args:
            config_path: curriculum_config.yaml 路径
            airsim_client: AirSim 客户端实例
            outcome_tracker: 用于共享的窗口统计器（确保 TB 与课程一致）
        """
        self.logger = logging.getLogger(__name__)
        self.airsim_client = airsim_client

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.enabled = self.config['curriculum_learning']['enabled']
        self.levels = self.config['curriculum_learning']['levels']

        # 共享的最近 20 场统计
        default_window = self.config['curriculum_learning'].get('window_size', 20)
        self.tracker = outcome_tracker if outcome_tracker is not None else RollingOutcomeStats(window_size=default_window)
        self.window_size = self.tracker.window_size

        # 当前等级
        self.current_level = 0
        self.current_level_idx = 0  # 兼容

        # 缓存当前等级的障碍命令
        self.current_obstacle_command = ""

        # 统计数据（课程判定用）
        self.episode_count = 0
        self.success_count = 0
        self.total_reward = 0.0

        self.logger.info(f"✅ CurriculumManager initialized: {len(self.levels)} levels")
        if self.enabled:
            self._prepare_level(0)  # 只准备参数，不生成

    def reset(self) -> Dict:
        """
        Episode 开始时调用
        - 每次都重新生成障碍物（实现随机性）
        - 使用当前等级的参数
        """
        if self.enabled:
            self.regenerate_obstacles()

        if not self.enabled:
            return {
                'level': 0,
                'level_name': 'Fixed',
                'progress': 1.0,
                **self.tracker.rates()
            }

        return {
            'level': self.current_level,
            'level_name': self.levels[self.current_level]['name'],
            'success_rate': self.success_count / max(self.episode_count, 1),
            'episode_count': self.episode_count,
            'total_levels': len(self.levels),
            **self.tracker.rates()
        }

    def regenerate_obstacles(self):
        """重新生成障碍物（使用当前等级的参数）"""
        if not self.current_obstacle_command:
            self.logger.warning("⚠️  No obstacle command prepared")
            return

        try:
            self.airsim_client.simRunConsoleCommand("ClearAllObstacles")
            time.sleep(0.2)
            self.airsim_client.simRunConsoleCommand(self.current_obstacle_command)
            time.sleep(0.5)

            if self.episode_count % 10 == 0:
                self.logger.info(f"🔄 Regenerated obstacles for Level {self.current_level + 1}")

        except Exception as e:
            self.logger.error(f"❌ Failed to regenerate obstacles: {e}")

    def update(
        self,
        done: bool,
        episode_success: bool,
        total_reward: float,
        done_info: Optional[Dict] = None
    ) -> Tuple[bool, Dict]:
        """
        每个 step 结束时调用，更新统计并检查是否晋级

        Args:
            done: episode 是否结束
            episode_success: 是否成功
            total_reward: 总奖励
            done_info: 环境 info，包含终止原因
        """
        if not self.enabled:
            info = self.get_current_level_info()
            info.update(self.tracker.rates())
            return False, info

        if not done:
            info = self.get_current_level_info()
            info.update(self.tracker.rates())
            return False, info

        outcome = categorize_outcome(done_info, episode_success)
        self.tracker.add(outcome)
        rates = self.tracker.rates()

        self.episode_count += 1
        if outcome == "success":
            self.success_count += 1
        self.total_reward += total_reward

        level_changed = False
        if self._should_level_up(rates):
            level_changed = self._level_up()

        info = self.get_current_level_info()
        info.update(rates)
        return level_changed, info

    def _should_level_up(self, rates: Optional[Dict[str, float]] = None) -> bool:
        """判断是否应该晋级（基于最近窗口的 success_rate_20）"""
        if self.current_level >= len(self.levels) - 1:
            return False

        current_config = self.levels[self.current_level]
        min_episodes = current_config.get('min_episodes', 20)

        if self.episode_count < min_episodes:
            return False

        rates = rates or self.tracker.rates()
        threshold = current_config.get('trigger_success_rate', 0.7)
        return rates.get("success_rate_20", 0.0) >= threshold

    def _level_up(self) -> bool:
        """执行晋级操作"""
        old_level = self.current_level
        self.current_level += 1
        self.current_level_idx = self.current_level

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎉 LEVEL UP! {old_level + 1} → {self.current_level + 1}")
        self.logger.info(f"📊 Statistics:")
        self.logger.info(f"   - Total episodes: {self.episode_count}")
        self.logger.info(f"   - Success rate (recent): {self.tracker.rates().get('success_rate_20', 0.0):.2%}")
        self.logger.info(f"   - Avg reward: {self.total_reward / max(self.episode_count, 1):.2f}")
        self.logger.info(f"{'='*60}\n")

        success = self._prepare_level(self.current_level)

        # 重置统计 & 最近窗口
        self.episode_count = 0
        self.success_count = 0
        self.total_reward = 0.0
        self.tracker.reset()

        return success

    def _prepare_level(self, level_idx: int) -> bool:
        """
        准备指定等级的障碍物参数（不生成障碍物）
        """
        if level_idx >= len(self.levels):
            self.logger.warning(f"Invalid level index: {level_idx}")
            return False

        level = self.levels[level_idx]
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📚 Preparing Level {level_idx + 1}: {level['name']}")
        self.logger.info(f"{'='*60}")

        obstacle_configs = []

        for obs_config in level['obstacles']:
            mesh_name = obs_config['mesh_name']
            count = obs_config['count']

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
                'episode_count': self.episode_count,
                **self.tracker.rates()
            }

        return {
            'name': self.levels[self.current_level]['name'],
            'level': self.current_level,
            'total_levels': len(self.levels),
            'success_rate': self.success_count / max(self.episode_count, 1),
            'episode_count': self.episode_count,
            'avg_reward': self.total_reward / max(self.episode_count, 1),
            **self.tracker.rates()
        }

    def force_level(self, level_idx: int):
        """强制切换到指定等级（调试用）"""
        if 0 <= level_idx < len(self.levels):
            self.current_level = level_idx
            self.current_level_idx = level_idx
            self.episode_count = 0
            self.success_count = 0
            self.total_reward = 0.0
            self.tracker.reset()
            self._prepare_level(level_idx)
        else:
            self.logger.error(f"Invalid level index: {level_idx}")
