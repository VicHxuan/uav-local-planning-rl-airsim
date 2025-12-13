# Reset 状态稳定性检测方案

## 需求分析

在每次 `reset()` 后，需要确保：
1. AirSim 仿真状态已完全稳定
2. 时间戳连续性
3. 传感器数据可用（激光雷达）
4. 位置/姿态数据一致性

## 问题诊断

### 当前 reset 流程的问题

**文件**: `airsim_env.py` Line 208-261 和 `multirotor_airsim.py` Line 80-105

**问题**：
1. `self.client.reset()` 后没有等待机制
2. `moveToZAsync().join()` 可能提前返回，但实际位置还未稳定
3. 没有验证传感器数据的可用性
4. 使用了固定的 `time.sleep(1)`，不够智能

**潜在风险**：
- 第一次 `get_position()` 返回过时数据
- 第一次 `getLidarData()` 可能返回空数据
- 时间戳可能不连续

---

## 解决方案设计

### 方案1：基于时间戳的状态验证（推荐）

创建一个状态稳定性检查函数，验证 AirSim 状态是否稳定。

**位置**: `airsim_env.py` 或 `multirotor_airsim.py`

**添加新方法**:

```python
def _wait_for_simulation_stable(self, max_wait_time=5.0, check_interval=0.1):
    """
    等待仿真状态稳定
    
    Args:
        max_wait_time: 最大等待时间（秒）
        check_interval: 检查间隔（秒）
    
    Returns:
        bool: 是否成功稳定
    """
    start_time = time.time()
    stable_count = 0
    required_stable_checks = 3  # 需要连续3次检查都稳定
    
    last_timestamp_of_state = None
    
    while time.time() - start_time < max_wait_time:
        try:
            # 1. 检查是否能获取状态
            states = self.client.getMultirotorState()
            
            # 2. 检查时间戳是否连续
            current_timestamp = states.timestamp  # AirSim 返回的时间戳
            
            if last_timestamp_of_state is not None:
                time_diff = current_timestamp - last_timestamp_of_state
                # 检查时间戳是否合理（应该是大约 check_interval 秒）
                if time_diff < 0 or time_diff > 2 * check_interval:
                    # 时间戳不连续或倒退，不稳定
                    stable_count = 0
                    last_timestamp_of_state = current_timestamp
                    time.sleep(check_interval)
                    continue
            
            last_timestamp_of_state = current_timestamp
            
            # 3. 检查速度是否接近0（起飞完成后应该是静止或接近静止）
            vel = states.kinematics_estimated.linear_velocity
            speed = math.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2)
            
            if speed > 0.5:  # 速度大于 0.5 m/s 认为不稳定
                stable_count = 0
                time.sleep(check_interval)
                continue
            
            # 4. 检查位置数据的合理性
            pose = self.client.simGetVehiclePose()
            position = pose.position
            
            # 检查位置是否在合理范围内
            if abs(position.x_val) > 1000 or abs(position.y_val) > 1000:
                # 位置异常，不稳定
                stable_count = 0
                time.sleep(check_interval)
                continue
            
            # 所有检查通过，计数器+1
            stable_count += 1
            
            if stable_count >= required_stable_checks:
                return True  # 连续N次检查都通过，认为稳定
            
        except Exception as e:
            # 如果在获取状态时出错，说明还不稳定
            print(f"[稳定性检查] 获取状态失败: {e}")
            stable_count = 0
        
        time.sleep(check_interval)
    
    print(f"[稳定性检查] 超时：在 {max_wait_time}s 内未达到稳定状态")
    return False
```

---

### 方案2：基于传感器数据验证的轻量级版本

如果不关心时间戳，只关心数据可用性：

```python
def _wait_for_simulation_stable_light(self, max_wait_time=3.0):
    """
    轻量级状态稳定性检查
    
    只需验证：
    1. 能成功获取位置
    2. 能成功获取激光雷达数据
    3. 连续几次获取的数据一致
    """
    start_time = time.time()
    stable_positions = []
    required_consistent_checks = 2
    
    while time.time() - start_time < max_wait_time:
        try:
            # 获取位置
            pose = self.client.simGetVehiclePose()
            current_pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
            
            stable_positions.append(current_pos)
            
            # 如果已经有足够多的位置数据
            if len(stable_positions) >= required_consistent_checks:
                # 检查位置是否稳定（变化小于阈值）
                recent_positions = stable_positions[-required_consistent_checks:]
                pos_variance = np.std(recent_positions, axis=0)
                
                if np.all(pos_variance < 0.1):  # 位置变化小于 0.1m
                    # 还会尝试获取一次激光雷达数据
                    try:
                        lidar_data = self.client.getLidarData(lidar_name="LidarSensor1")
                        if len(lidar_data.point_cloud) > 0:
                            return True
                    except Exception as e:
                        print(f"[轻量检查] 激光雷达数据获取失败: {e}")
                
                # 移除最旧的位置
                stable_positions.pop(0)
        
        except Exception as e:
            print(f"[轻量检查] 状态获取失败: {e}")
        
        time.sleep(0.1)
    
    print(f"[轻量检查] 超时，可能状态不稳定")
    return False
```

---

### 方案3：针对 reset 后的完整验证流程

完整的 reset 后稳定性检查，包含所有关键验证点：

```python
def verify_post_reset_stability(self):
    """
    reset 后的完整验证流程
    
    验证清单：
    1. API 连接正常
    2. 位置数据合理
    3. 速度接近0
    4. 传感器可用
    5. 碰撞信息可用
    6. 激光雷达数据可用（可选）
    """
    verification_results = {
        'api_connected': False,
        'position_valid': False,
        'velocity_stable': False,
        'sensors_ready': False,
        'collision_system_ready': False,
        'lidar_available': False,
    }
    
    try:
        # 1. 验证 API 连接
        states = self.client.getMultirotorState()
        verification_results['api_connected'] = True
        
        # 2. 验证位置数据
        pose = self.client.simGetVehiclePose()
        position = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        
        if all(-1000 < p < 1000 for p in position):  # 位置在合理范围内
            verification_results['position_valid'] = True
        
        # 3. 验证速度稳定性
        vel = states.kinematics_estimated.linear_velocity
        speed = math.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2)
        
        if speed < 1.0:  # 速度小于 1 m/s
            verification_results['velocity_stable'] = True
        
        # 4. 验证传感器系统
        states_timestamp = states.timestamp
        if states_timestamp is not None and states_timestamp > 0:
            verification_results['sensors_ready'] = True
        
        # 5. 验证碰撞系统
        try:
            collision_info = self.client.simGetCollisionInfo()
            verification_results['collision_system_ready'] = True
        except:
            pass
        
        # 6. 验证激光雷达（可选，因为可能需要时间初始化）
        try:
            lidar_data = self.client.getLidarData(lidar_name=getattr(self, 'lidar_sensor_name', 'LidarSensor1'))
            if lidar_data is not None:
                verification_results['lidar_available'] = True
        except:
            pass
        
    except Exception as e:
        print(f"[验证失败] {e}")
    
    # 打印验证结果
    print(f"[Reset验证] API:{verification_results['api_connected']} "
          f"Pos:{verification_results['position_valid']} "
          f"Vel:{verification_results['velocity_stable']} "
          f"Sensors:{verification_results['sensors_ready']} "
          f"Collision:{verification_results['collision_system_ready']} "
          f"Lidar:{verification_results['lidar_available']}")
    
    return verification_results
```

---

## 集成到现有代码

### 修改1：在 multiorotor_airsim.py 的 reset() 方法中

**位置**: `multirotor_airsim.py` Line 80-105

**修改建议**:

```python
def reset(self):
    self.client.reset()
    
    # ========== 新增：等待 reset 完成 ==========
    time.sleep(0.5)  # 给 AirSim 一些时间完成 reset
    
    # ========== 现有代码 ==========
    self.init_goal_pose_2D()
    self.goal_move_direction = np.random.uniform(0, 2 * math.pi)
    print(f'Goal move direction initialized: {math.degrees(self.goal_move_direction):.1f} degrees')
    print('New goal pose: ', self.goal_position)
    
    # reset start
    yaw_noise = self.start_random_angle * np.random.random()
    # set airsim pose
    pose = self.client.simGetVehiclePose()
    pose.position.x_val = self.start_position[0]
    pose.position.y_val = self.start_position[1]
    pose.position.z_val = - self.start_position[2]
    print('Start pose: ', pose.position.x_val, pose.position.y_val, -pose.position.z_val)
    pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
    self.client.simSetVehiclePose(pose, True)

    self.client.simPause(False)
    self.client.enableApiControl(563)
    self.client.armDisarm(True)

    # take off
    self.client.moveToZAsync(-self.start_position[2], 2).join()
    
    # ========== 新增：等待起飞完成且状态稳定 ==========
    # 方案1：基于时间戳的完整检查
    if not self._wait_for_simulation_stable(max_wait_time=3.0):
        print("[警告] 状态稳定性检查失败，继续执行...")
    
    # 或者方案2：轻量级检查
    # if not self._wait_for_simulation_stable_light(max_wait_time=2.0):
    #     print("[警告] 轻量级检查失败，继续执行...")
    
    self.client.simPause(True)
```

---

### 修改2：在 airsim_env.py 的 reset() 方法中

**位置**: `airsim_env.py` Line 208 vial 261

**修改建议**:

```python
def reset(self):
    # reset state
    self.dynamic_model.reset()
    
    # ========== 新增：验证 reset 后的状态 ==========
    verification = self.dynamic_model.verify_post_reset_stability()
    if not verification['position_valid']:
        print("[严重警告] Reset 后位置数据异常！")
    if not verification['velocity_stable']:
        print("[警告] Reset 后速度不稳定，可能需要更多时间")
    
    # ========== 现有代码 ==========
    if self.step_num != 0:
        cmd = "ClearAllObstacles"
        time.sleep(1)
        cmd = "SpawnObstacles"
        time.sleep(1)
        self.dynamic_model.client.simRunConsoleCommand(cmd)
        
    self.episode_num += 1
    self.step_num = 0
    
    # ... 其余代码保持不变 ...
    
    # ========== 新增：在获取观测前验证 ==========
    # 确保状态和历史数据已初始化
    if len(self._pos_hist) < 2:
        print("[警告] 位置历史未正确初始化")
    if len(self._vel_hist) < 2:
        print("[警告] 速度历史未正确初始化")
    
    # ... 继续现有代码 ...
    obs = self.get_obs()
    return obs
```

---

## 配置选项

可以在配置文件中添加稳定性检查参数：

**配置文件** `xxx.ini`:

```ini
[environment]
# 其他配置...

# Reset 稳定性检查参数
reset_stability_check_enabled = True
reset_stability_max_wait_time = 3.0
reset_stability_check_interval = 0.1
reset_stability_speed_threshold = 0.5
```

**在代码中读取**:

```python
# 在 multiorotor_airsim.py 的 __init__ 或 reset 中
try:
    self.stability_check_enabled = cfg.getboolean('environment', 'reset_stability_check_enabled', fallback=True)
    self.stability_max_wait = cfg.getfloat('environment', 'reset_stability_max_wait_time', fallback=3.0)
    self.stability_speed_threshold = cfg.getfloat('environment', 'reset_stability_speed_threshold', fallback=0.5)
except:
    # 使用默认值
    self.stability_check_enabled = True
    self.stability_max_wait = 3.0
    self.stability_speed_threshold = 0.5
```

---

## 实施建议

### 优先级

1. **立即实施**：方案2（轻量级检查）- 最简单，风险低
2. **本周内**：方案1（完整检查）- 如果需要更强的一致性保证
3. **测试阶段**：方案3（完整验证）- 用于深度调试

### 性能考量

- 方案2（轻量级）: 平均等待 0.2-0.5 秒
- 方案1（完整）: 平均等待 0.5-2 秒
- 方案3（验证）: 零等待，只做检查

### 建议的组合使用

```python
# 1. 先做验证（快，不等待）
verification = self.verify_post_reset_stability()

# 2. 如果验证失败，再做稳定性检查（慢，但确保稳定）
if not verification['velocity_stable']:
    self._wait_for_simulation_stable_light(max_wait_time=2.0)
```

---

## 测试方法

添加一个测试函数：

```python
def test_reset_stability(self):
    """测试 reset 稳定性检查"""
    print("测试 Reset 稳定性...")
    
    for i in range(5):
        print(f"\n--- 第 {i+1} 次 Reset ---")
        self.reset()
        
        verification = self.verify_post_reset_stability()
        
        all_ok = all(verification.values())
        print(f"结果: {'通过' if all_ok else 'Evaluation'}")
        
        time.sleep(0.5)
```

---

## 注意事项

1. **时间戳格式**：AirSim 的时间戳可能是纳秒或秒，需要确认 `states.timestamp` 的单位
2. **等待时间**：如果系统较慢，可能需要增加 `max_wait_time`
3. **激光雷达**：初始化可能需要额外时间，建议作为非必要验证项
4. **性能影响**：稳定检查会增加 reset 时间，权衡稳定性和训练速度

---

## 预期效果

### 修改前
```python
reset()  # 可能立即返回，但数据不稳定
get_position()  # 可能返回异常值
```

### 修改后
```python
reset()  # 等待状态稳定后才返回
verify_post_reset_stability()  # 通过所有检查
get_position()  # 返回稳定且正确的值
```

---

## 故障排查

如果稳定检查总是失败：

1. **增加等待时间**: `max_wait_time` 从 3.0 增加到 5.0
2. **放宽阈值**: 速度阈值从 0.5 增加到 1.0
3. **禁用检查**: 暂时设置 `stability_check_enabled = False`
4. **添加日志**: 在每个检查点打印详细信息



























































