"""AirSim connection smoke test."""

import airsim
import time
import math
import numpy as np

def main():
    # 1. 连接到 AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("----------------------------------------")
    print("[1/4] Taking off...")
    client.takeoffAsync().join()

    # 2. 升高到 5m (注意 NED 坐标系，高度为负值)
    target_z = -5.0
    print(f"[2/4] Ascending to {abs(target_z)}m...")
    # 速度设为 2 m/s
    client.moveToZAsync(target_z, 2).join()

    # 3. 定义目标参数
    # 目标偏航角 (度)
    target_yaw_deg = 30.0
    target_yaw_rad = math.radians(target_yaw_deg)
    
    # 目标机体速度 (Body Frame)
    # vx_body = 0 (不向前)
    # vy_body = 1 (向右平移/侧飞)
    v_body_x = 0.0
    v_body_y = 1.0 
    
    print(f"[3/4] calculating velocity components...")
    print(f"      Target Yaw: {target_yaw_deg} deg")
    print(f"      Target Body Vy: {v_body_y} m/s")

    # --- [关键步骤] 将机体速度转换为世界坐标系速度 (NED) ---
    # 公式: V_world = R_z * V_body
    # R_z = [[cos, -sin], [sin, cos]]
    
    vx_world = v_body_x * math.cos(target_yaw_rad) - v_body_y * math.sin(target_yaw_rad)
    vy_world = v_body_x * math.sin(target_yaw_rad) + v_body_y * math.cos(target_yaw_rad)

    print(f"      Calculated World Vx: {vx_world:.2f} m/s")
    print(f"      Calculated World Vy: {vy_world:.2f} m/s")
    
    # 4. 持续发布指令
    print("[4/4] Executing Maneuver (Duration: 10s)...")
    
    duration = 10.0
    start_time = time.time()
    
    # 使用循环发送指令，模拟控制回路
    while time.time() - start_time < duration:
        client.moveByVelocityZAsync(
            vx=vx_world,
            vy=vy_world,
            z=target_z,        # 保持高度 -5m
            duration=0.1,      # 每次指令有效时间短一点，保证平滑
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, # 关键：允许 偏航 与 移动方向 分离
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=target_yaw_deg) # 固定偏航角模式
        ).join()

    # 5. 悬停并着陆
    print("Maneuver complete. Hovering...")
    client.moveByVelocityAsync(0, 0, 0, 2).join() # 刹车
    
    # 如果想落地，取消下面注释
    # print("Landing...")
    # client.landAsync().join()
    # client.armDisarm(False)
    # client.enableApiControl(False)

if __name__ == "__main__":
    main()
