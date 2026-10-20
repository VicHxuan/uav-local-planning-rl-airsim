"""Training entry with live GUI plots."""

import sys
import torch
import argparse
import numpy as np
from PyQt5 import QtWidgets
from utils.ui_train import TrainingUi
from configparser import ConfigParser
from utils.thread_train import TrainingThread


def main():

    config_file = 'configs/NavRL2D_SAC-Transformer_encoder-vector_lidar_multi-EGOreward.ini'
    # 1. Create the qt thread
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config_file)
    gui.show()

    # 2. Start training thread
    training_thread = TrainingThread(config_file)

    # 【修改】将 env 改为 r_env (raw env)，因为 env 现在是 VecWrapper，没有信号属性
    training_thread.r_env.action_signal.connect(gui.action_cb)
    training_thread.r_env.state_signal.connect(gui.state_cb)
    training_thread.r_env.attitude_signal.connect(gui.attitude_plot_cb)
    training_thread.r_env.reward_signal.connect(gui.reward_plot_cb)
    training_thread.r_env.pose_signal.connect(gui.traj_plot_cb)

    cfg = ConfigParser()
    cfg.read(config_file)

    training_thread.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
