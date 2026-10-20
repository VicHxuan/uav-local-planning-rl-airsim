import random
import numpy as np
import torch

import sys
import argparse
from PyQt5 import QtWidgets

from utils.thread_evaluation import EvaluateThread
from utils.ui_train import TrainingUi
from configparser import ConfigParser


def get_parser():
    # 保持原样
    parser = argparse.ArgumentParser(description="trained model evaluation with plot")
    parser.add_argument('-model_path', required=True, help='model path')
    parser.add_argument('-eval_eps', required=True, type=int, help='evaluation episode number')
    return parser


def main():

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Random seeds reset to {SEED}")

    # 请根据实际情况修改路径
    eval_path = r'/home/user/Transformer_Trajectory/logs/Easy_2D/2025_12_14_21_18_seed1070139711_Multirotor_MLP_embedding_Transformer_encoder_SAC'
    config_file = eval_path + '/config/config.ini'
    print(config_file)
    model_file = eval_path + '/models/model_sb3_600000_steps.zip'
    total_eval_episodes = 50

    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config=config_file)
    gui.show()

    evaluate_thread = EvaluateThread(
        eval_path, config_file, model_file, total_eval_episodes)
    
    # 【修改】使用 r_env (raw env) 连接信号
    evaluate_thread.r_env.action_signal.connect(gui.action_cb)
    evaluate_thread.r_env.state_signal.connect(gui.state_cb)
    evaluate_thread.r_env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.r_env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.r_env.pose_signal.connect(gui.traj_plot_cb)

    cfg = ConfigParser()
    cfg.read(config_file)
    if cfg.has_option('options', 'perception'):
        if cfg.get('options', 'perception') == 'lgmd':
            evaluate_thread.r_env.lgmd_signal.connect(gui.lgmd_plot_cb)

    evaluate_thread.start()

    sys.exit(app.exec_())
    print('Exiting program')


if __name__ == "__main__":
    main()
