#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal SB3 inference (no environment).
- Loads a trained SB3 model (SAC/TD3/PPO).
- Reads observation from stdin and prints the action.
- Works with custom feature extractors by importing your module.
Usage:
  python infer_sb3_min.py --model /path/to/model_sb3.zip [--algo SAC|TD3|PPO] [--deterministic]
Commands in the REPL:
  - Enter numbers separated by space/comma. Length must equal np.prod(observation_space.shape).
  - Or paste a JSON-like array (e.g., [0.1, 0.2, ...]).
  - Special inputs: zeros, ones, randn, help, shape, reset, quit/exit.
"""
import os
import sys
import json
import argparse
import numpy as np

# Ensure we can import the custom feature extractor module in the same folder.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

# Import your custom feature extractors so SB3 can unpickle the model.
# Even if they are not referenced directly here, importing the module ensures the classes
# (e.g., Raw / MLP / LSTM / LSTM_attention / CNN_GAP) are available.
try:
    import utils.custom_policy_sb3  # noqa: F401
except Exception as e:
    print("[WARN] 无法导入 custom_policy_sb3（如果模型使用了自定义特征提取器，load 会失败）:", e)

from stable_baselines3 import SAC, TD3, PPO

ALGOS = {
    "SAC": SAC,
    "TD3": TD3,
    "PPO": PPO,
}

def auto_load(model_path: str):
    """Try to load the model with SAC, TD3, then PPO."""
    last_err = None
    for name, cls in ALGOS.items():
        try:
            print(f"[INFO] 尝试以 {name} 加载模型...")
            m = cls.load(model_path, device="auto")
            print(f"[OK] 成功以 {name} 加载。")
            return name, m
        except Exception as e:
            last_err = e
            print(f"[INFO] 以 {name} 加载失败：{e}")
    raise RuntimeError(f"无法识别/加载模型：{last_err}")

def parse_args():
    p = argparse.ArgumentParser("SB3 极简推理（无环境）")
    p.add_argument("--model", required=True, help="训练好的模型路径，如 ./models/model_sb3.zip")
    p.add_argument("--algo", choices=list(ALGOS.keys()), help="显式指定算法；若不指定则自动尝试")
    p.add_argument("--deterministic", action="store_true", help="确定性策略（默认 False）")
    return p.parse_args()

def print_obs_spec(model):
    try:
        space = model.observation_space
    except Exception:
        # Some very old/edge saves might not include observation_space.
        print("[WARN] 模型未携带 observation_space，无法自动推断形状，请手工确保输入与训练一致。")
        return None
    print("[INFO] 观测空间：", type(space).__name__)
    if hasattr(space, "shape") and space.shape is not None:
        print("[INFO] 观测 shape =", space.shape, " (size =", int(np.prod(space.shape)), ")")
    if hasattr(space, "dtype"):
        print("[INFO] 观测 dtype  =", space.dtype)
    return space

def to_array(s, space):
    """Parse a line of text into an observation numpy array shaped as space.shape."""
    s = s.strip()
    if s.lower() in ("quit", "exit", "q"):
        return "QUIT"
    if s.lower() == "help":
        print(__doc__)
        return None
    if s.lower() == "shape":
        print_obs_spec(model)
        return None
    if s.lower() == "reset":
        # Best-effort reset for recurrent/custom extractors (if implemented)
        fx = getattr(model.policy, "features_extractor", None)
        if fx is not None:
            for name in ("reset", "reset_states", "reset_hidden", "reset_lstm"):
                if hasattr(fx, name) and callable(getattr(fx, name)):
                    try:
                        getattr(fx, name)()
                        print("[OK] 已调用 features_extractor.%s()" % name)
                        break
                    except Exception as e:
                        print("[WARN] 重置失败：", e)
                        break
        else:
            print("[INFO] 无 features_extractor 可重置。")
        return None

    size = None
    shape = getattr(space, "shape", None)
    dtype = getattr(space, "dtype", np.float32)

    if shape is not None:
        size = int(np.prod(shape))

    # shortcuts
    if s.lower() == "zeros":
        arr = np.zeros(size, dtype=dtype)
    elif s.lower() == "ones":
        arr = np.ones(size, dtype=dtype)
    elif s.lower() == "randn":
        arr = np.random.randn(size).astype(dtype)
    else:
        # Try JSON first
        try:
            obj = json.loads(s)
            arr = np.array(obj, dtype=dtype)
        except Exception:
            # Fallback: split by comma/space
            tokens = [t for t in s.replace(",", " ").split() if t]
            arr = np.array([float(t) for t in tokens], dtype=dtype)

    if shape is not None:
        if arr.size != size:
            print(f"[ERR] 输入元素数量={arr.size} 与期望={size} 不一致（期望形状 {shape}）。")
            return None
        arr = arr.reshape(shape)
    return arr

if __name__ == "__main__":
    args = parse_args()
    if args.algo:
        Algo = ALGOS[args.algo.upper()]
        model = Algo.load(args.model, device="auto")
        algo_name = args.algo.upper()
        print(f"[OK] 以 {algo_name} 加载模型。")
    else:
        algo_name, model = auto_load(args.model)

    space = print_obs_spec(model)

    print("\n[READY] 请输入观测（或输入 help 查看帮助；输入 quit 退出）")
    while True:
        try:
            line = input("obs> ")
        except EOFError:
            break
        obs = to_array(line, space)
        if obs is None:
            continue
        if obs == "QUIT":
            break
        try:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            # Convert to plain list for readability
            if isinstance(action, np.ndarray):
                out = action.tolist()
            else:
                # Some algos might return scalar or list
                out = action
            print("action =", out)
        except Exception as e:
            print("[ERR] 预测失败：", e)
            continue
