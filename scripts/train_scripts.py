import sys
import torch
import argparse
import numpy as np
from PyQt5 import QtWidgets
from utils.ui_set import TrainingUi
from configparser import ConfigParser
from utils.thread_train import TrainingThread

def main():
    config_file = 'configs/test.ini'
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config_file)
    gui.show()
    training_thread = TrainingThread(config_file)
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
        pass
