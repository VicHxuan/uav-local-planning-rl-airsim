# 终端输出重构方案

## 问题诊断

### 当前问题
1. **双重输出**：同时使用 `print()` 和 `simPrintLogMessage()`，导致重复输出
2. **无日志级别**：所有信息都同等重要，无法筛选
3. **调试信息混杂**：正式运行时仍输出大量调试信息
4. **eye-catcher过载**：你添加的"开始打印"、"结束打印"标记太冗长
5. **每步都打印**：`print_train_info_airsim()` 每个 step 都调用（line 315）

---

## 修改建议

### 修改1：在文件开头添加日志配置

**位置**：`airsim_env.py` 第1-16行（import区域之后）

```python
# 添加统一的日志系统
import logging
from enum import Enum

class LogLevel(Enum):
    """日志级别枚举"""
    SILENT = 0      # 静默模式
    MINIMAL = 1     # 仅关键信息
    NORMAL = 2      # 正常模式
    VERBOSE = 3     # 详细模式
    DEBUG = 4       # 调试模式

# 全局日志级别（从配置文件读取或默认）
ENV_LOG_LEVEL = LogLevel.NORMAL

def env_log(level: LogLevel, message: str, category: str = "INFO"):
    """统一的日志输出函数"""
    if level.value <= ENV_LOG_LEVEL.value:
        prefix = f"[{category}]"
        print(f"{prefix} {message}")
```

---

### 修改2：替换 multorotor_airsim.py 中的所有 print

**文件**：`multirotor_airsim.py`

**Line 156 & 166**：目标初始化信息
```python
# 原代码
print("r,phi,dx,dy",self.r,phi,dx,dy)
print("goal_x,goal_y,goal_z",goal_x,goal_y,goal_z)

# 改为（从 airsim_env 导入 env_log）
env_log(LogLevel.VERBOSE, f"Goal init: r={self.r:.2f}, phi={phi:.2f}, dx={dx:.2f}, dy={dy:.2f}", "GOAL")
env_log(LogLevel.VERBOSE, f"Goal position: ({goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f})", "GOAL")
```

**Line 84, 86, 94**：Reset 信息
```python
# 原代码
print(f'Goal move direction initialized: {math.degrees(self.goal_move_direction):.1f} degrees')
print('New goal pose: ', self.goal_position)
print('Start pose: ', pose.position.x_val, pose.position.y_val, -pose.position.z_val)

# 改为
env_log(LogLevel.NORMAL, f"Episode reset - Goal dir: {math.degrees(self.goal_move_direction):.1f}°", "RESET")
env_log(LogLevel.VERBOSE, f"Goal pose: {self.goal_position}", "RESET")
env_log(LogLevel.VERBOSE, f"Start pose: ({pose.position.x_val:.2f}, {pose.position.y_val:.2f}, {-pose.position.z_val:.2f})", "RESET")
```

---

### 修改3：重构 airsim_env.py 的打印

**Line 315-320**：只在关键点打印
```python
# 原代码（每步都打印）
self.print_train_info_airsim(action, obs, reward, info)

# 改为（仅在 episode 结束时或每 N 步打印）
if done or (self.step_num % 100 == 0):  # 每100步或episode结束时打印
    self.print_train_info_airsim(action, obs, reward, info)
```

**Line 290-292**：简化调试标记
```python
# 原代码
if done:
    print('done_info_开始打印')
    print(info)
    print('done_info_结束打印')

# 改为
if done:
    env_log(LogLevel.NORMAL, f"Episode terminated: {info}", "DONE")
```

**Line 330-341**：简化 keyboard_debug 输出
```python
# 原代码（太冗长）
print('step_info_开始打印')
print('=============================================================================')
print('episode', self.episode_num, 'step', ...)
print('step_info_结束打印')

# 改为
env_log(LogLevel.DEBUG, "=" * 80, "DEBUG")
env_log(LogLevel.DEBUG, f"EP:{self.episode_num} Step:{self.step_num} Total:{self.total_step}", "DEBUG")
env_log(LogLevel.DEBUG, f"Action: {action_copy} | State: {state_copy}", "DEBUG")
env_log(LogLevel.DEBUG, f"Reward: {reward:.3f} ({self.cumulated_episode_reward:.3f}) | Done: {done}", "DEBUG")
```

**Line 706-711**：超出工作空间
```python
# 原代码
print('超出工作空间info_开始打印')
print(f"current_position: {current_position}")
...

# 改为
env_log(LogLevel.WARNING, f"Out of workspace! Position: {current_position}", "WORKSPACE")
env_log(LogLevel.VERBOSE, f"Workspace limits: X{self.work_space_x}, Y{self.work_space_y}, Z{self.work_space_z}", "WORKSPACE")
```

**Line 733-739**：碰撞检测
```python
# 原代码
print('碰撞info_开始打印')
print(f"min_distance_to_obstacles: {self.min_distance_to_obstacles}")
...

# 改为
env_log(LogLevel.WARNING, f"Collision detected! Min_dist: {self.min_distance_to_obstacles:.2f}m", "CRASH")
env_log(LogLevel.VERBOSE, f"Crash details: dist<{self.crash_distance}m, collision={collision_info.has_collided}", "CRASH")
```

---

### 修改4：简化 simPrintLogMessage 调用

**文件**：`airsim_env.py` line 805-820

**方案**：只在 Verbose 模式下输出到 AirSim

```python
def print_train_info_airsim(self, action, obs, reward, info):
    # 控制台输出
    env_log(LogLevel.NORMAL, f"EP:{self.episode_num} S:{self.step_num} | Reward: {reward:.3f}", "STEP")
    
    # AirSim UI 输出（可选，仅调试时）
    if ENV_LOG_LEVEL.value >= LogLevel.VERBOSE.value:
        msg_train_info = "EP: {} Step: {} Total_step: {}".format(
            self.episode_num, self.step_num, self.total_step)
        self.client.simPrintLogMessage('Train: ', msg_train_info)
        # ... 其他 simPrintLogMessage 调用
```

---

### 修改5：配置文件支持

**文件**：配置文件（例如 `Easy2D_SAC-Transformer_encoder-vector_lidar_multi-EGOreward.ini`）

添加新选项：
```ini
[options]
# ... 现有配置
# 新增日志级别配置
log_level = NORMAL  # 可选: SILENT, MINIMAL, NORMAL, VERBOSE, DEBUG
```

**文件**：`airsim_env.py` line 57-65

```python
def set_config(self, cfg):
    """get config from .ini file"""
    global ENV_LOG_LEVEL
    
    # ... 现有代码
    
    # 读取日志级别
    try:
        log_level_str = cfg.get('options', 'log_level', fallback='NORMAL')
        ENV_LOG_LEVEL = LogLevel[log_level_str.upper()]
        env_log(LogLevel.NORMAL, f"Log level set to: {ENV_LOG_LEVEL.name}", "CONFIG")
    except (NoOptionError, KeyError):
        ENV_LOG_LEVEL = LogLevel.NORMAL
    
    # ... 继续现有代码
```

---

## 预期效果

### 正常运行（NORMAL 级别）
```
[INFO] init airsim-gym-env.
[INFO] Environment: Easy_2D Dynamics: Multirotor Perception: lidar_vector_multi
[INFO] Action smoothing enabled: True, window: 4
[RESET] Episode reset - Goal dir: 227.6°
[STEP] EP:1 S:0 | Reward: 0.245
[STEP] EP:1 S:100 | Reward: 1.523
[DONE] Episode terminated: {'is_success': False, ...}
```

### 调试模式（DEBUG 级别）
```
[INFO] init airsim-gym-env.
[DEBUG] =============================================================================
[DEBUG] EP:1 Step:0 Total:0
[DEBUG] Action: [0.5, 0.1] | State: [10.2, 45.3, 0.5, ...]
[GOAL] Goal init: r=58.87, phi=2.05, dx=-27.18, dy=52.21
[CRASH] Collision detected! Min_dist: 0.45m
```

---

## 修改优先级

1. **立即修改**：移除冗余的"开始打印"、"结束打印"标记
2. **立即修改**：减少 `print_train_info_airsim()` 调用频率（每100步而非每步）
3. **本周内**：统一所有 print 为 env_log
4. **可选**：添加配置文件支持（如果不着急）

---

## 注意事项

- 不要删除 `simPrintLogMessage`，它用于 AirSim UI 显示（但可以降频）
- 保留现有的错误捕获（line 437 的 LiDAR error）
- keyboard_debug 的输出应该保持详细（因为只在手动触发时使用）



























































