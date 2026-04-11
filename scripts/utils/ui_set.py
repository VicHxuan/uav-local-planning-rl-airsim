import sys
import math
from turtle import pen
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PIL import Image
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QWidget
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
import seaborn as sns
from configparser import ConfigParser

class TrainingUi(QWidget):
    def __init__(self, config):
        super(TrainingUi, self).__init__()
        self.cfg = ConfigParser()
        self.cfg.read(config)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Training UI")
        pg.setConfigOptions(leftButtonPan=False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.max_len = 100
        self.ptr = -self.max_len
        self.dynamics = self.cfg.get('options', 'dynamic_name')
        action_plot_gb = self.create_actionPlot_groupBox_multirotor()
        state_plot_gb = self.create_state_plot_groupbox()
        attitude_plot_gb = self.create_attitude_plot_groupbox()
        reward_plot_gb = self.create_reward_plot_groupbox()
        lgmd_plot_gb = self.create_lgmd_plot_groupbox()
        traj_plot_gb = self.create_traj_plot_groupbox()
        right_widget = QWidget()
        vlayout = QVBoxLayout()
        vlayout.addWidget(reward_plot_gb)
        if self.cfg.has_option('options', 'perception'):
            if self.cfg.get('options', 'perception') == 'lgmd':
                vlayout.addWidget(lgmd_plot_gb)
        vlayout.addWidget(traj_plot_gb)
        right_widget.setLayout(vlayout)
        main_layout = QHBoxLayout()
        main_layout.addWidget(action_plot_gb)
        main_layout.addWidget(state_plot_gb)
        main_layout.addWidget(attitude_plot_gb)
        main_layout.addWidget(right_widget)
        self.setLayout(main_layout)
        self.pen_red = pg.mkPen(color='r', width=2)
        self.pen_blue = pg.mkPen(color='b', width=1)
        self.pen_green = pg.mkPen(color='g', width=2)

    def update_value_list(self, list_data, value):
        list_data[:-1] = list_data[1:]
        list_data[-1] = float(value)
        return list_data

    def create_actionPlot_groupBox_multirotor(self):
        actionPlotGroupBox = QGroupBox('Action (multirotor)')
        self.v_xy_cmd_list = np.full(self.max_len, np.nan)
        self.v_xy_real_list = np.full(self.max_len, np.nan)
        self.v_z_cmd_list = np.full(self.max_len, np.nan)
        self.v_z_real_list = np.full(self.max_len, np.nan)
        self.yaw_rate_cmd_list = np.full(self.max_len, np.nan)
        self.yaw_rate_list = np.full(self.max_len, np.nan)
        layout = QVBoxLayout()
        self.plotWidget_v_xy = pg.PlotWidget(title='v_xy (m/s)')
        self.plotWidget_v_xy.setYRange(max=self.cfg.getfloat('multirotor', 'v_xy_max'), min=-self.cfg.getfloat('multirotor', 'v_xy_max'))
        self.plotWidget_v_xy.showGrid(x=True, y=True)
        self.plot_v_xy = self.plotWidget_v_xy.plot()
        self.plot_v_xy_cmd = self.plotWidget_v_xy.plot()
        self.plotWidget_v_z = pg.PlotWidget(title='v_z (m/s)')
        self.plotWidget_v_z.setYRange(max=self.cfg.getfloat('multirotor', 'v_z_max'), min=-self.cfg.getfloat('multirotor', 'v_z_max'))
        self.plotWidget_v_z.showGrid(x=True, y=True)
        self.plot_v_z = self.plotWidget_v_z.plot()
        self.plot_v_z_cmd = self.plotWidget_v_z.plot()
        self.plotWidget_yaw_rate = pg.PlotWidget(title='yaw_rate (deg/s)')
        self.plotWidget_yaw_rate.setYRange(max=self.cfg.getfloat('multirotor', 'yaw_rate_max_deg'), min=-self.cfg.getfloat('multirotor', 'yaw_rate_max_deg'))
        self.plotWidget_yaw_rate.showGrid(x=True, y=True)
        self.plot_yaw_rate = self.plotWidget_yaw_rate.plot()
        self.plot_yaw_rate_cmd = self.plotWidget_yaw_rate.plot()
        layout.addWidget(self.plotWidget_v_xy)
        layout.addWidget(self.plotWidget_v_z)
        layout.addWidget(self.plotWidget_yaw_rate)
        actionPlotGroupBox.setLayout(layout)
        return actionPlotGroupBox

    def action_cb(self, step, action):
        self.action_cb_multirotor(step, action)

    def action_cb_multirotor(self, step, action):
        self.update_value_list(self.v_xy_cmd_list, action[0])
        self.update_value_list(self.v_z_cmd_list, action[1])
        self.update_value_list(self.yaw_rate_cmd_list, math.degrees(action[2]))
        self.plot_v_xy_cmd.setData(self.v_xy_cmd_list, pen=self.pen_blue)
        self.plot_v_z_cmd.setData(self.v_z_cmd_list, pen=self.pen_blue)
        self.plot_yaw_rate_cmd.setData(self.yaw_rate_cmd_list, pen=self.pen_blue)

    def action_cb_fixed_wing(self, step, action):
        self.update_value_list(self.v_xy_cmd_list, action[0])
        self.update_value_list(self.v_z_cmd_list, action[1])
        self.update_value_list(self.roll_cmd_list, action[2])
        self.plot_v_xy_cmd.setData(self.v_xy_cmd_list, pen=self.pen_blue)
        self.plot_v_z_cmd.setData(self.v_z_cmd_list, pen=self.pen_blue)
        self.plot_roll_cmd_a.setData(self.roll_cmd_list, pen=self.pen_blue)

    def create_state_plot_groupbox(self):
        state_plot_groupbox = QGroupBox(title='State feature (Raw)')
        self.distance_list = np.full(self.max_len, np.nan)
        self.vertical_dis_list = np.full(self.max_len, np.nan)
        self.relative_yaw_list = np.full(self.max_len, np.nan)
        layout = QVBoxLayout()
        self.pw1 = pg.PlotWidget(title='distance_xy (m)')
        self.pw1.showGrid(x=True, y=True)
        self.p1 = self.pw1.plot()
        self.pw2 = pg.PlotWidget(title='distance_z (m)')
        self.pw2.showGrid(x=True, y=True)
        self.p2 = self.pw2.plot()
        self.pw3 = pg.PlotWidget(title='relative yaw (deg)')
        self.pw3.showGrid(x=True, y=True)
        self.p3 = self.pw3.plot()
        layout.addWidget(self.pw1)
        layout.addWidget(self.pw2)
        layout.addWidget(self.pw3)
        state_plot_groupbox.setLayout(layout)
        return state_plot_groupbox

    def state_cb(self, step, state_raw):
        self.update_value_list(self.distance_list, state_raw[0])
        self.update_value_list(self.vertical_dis_list, state_raw[1])
        self.update_value_list(self.relative_yaw_list, state_raw[2])
        self.p1.setData(self.distance_list, pen=self.pen_red)
        self.p2.setData(self.vertical_dis_list, pen=self.pen_red)
        self.p3.setData(self.relative_yaw_list, pen=self.pen_red)
        self.update_value_list(self.v_xy_real_list, state_raw[3])
        self.update_value_list(self.v_z_real_list, state_raw[4])
        self.plot_v_xy.setData(self.v_xy_real_list, pen=self.pen_red)
        self.plot_v_z.setData(self.v_z_real_list, pen=self.pen_red)
        if self.dynamics == 'SimpleFixedwing':
            self.update_value_list(self.roll_list_a, state_raw[5])
            self.plot_roll_a.setData(self.roll_list_a, pen=self.pen_red)
        else:
            self.update_value_list(self.yaw_rate_list, state_raw[5])
            self.plot_yaw_rate.setData(self.yaw_rate_list, pen=self.pen_red)

    def create_attitude_plot_groupbox(self):
        plot_gb = QGroupBox(title='Attitude (Degrees)')
        layout = QVBoxLayout()
        self.roll_list = np.full(self.max_len, np.nan)
        self.roll_cmd_list = np.full(self.max_len, np.nan)
        self.pitch_list = np.full(self.max_len, np.nan)
        self.pitch_cmd_list = np.full(self.max_len, np.nan)
        self.yaw_list = np.full(self.max_len, np.nan)
        self.yaw_cmd_list = np.full(self.max_len, np.nan)
        self.pw_roll = pg.PlotWidget(title='roll (deg)')
        self.pw_roll.setYRange(max=45, min=-45)
        self.pw_roll.showGrid(x=True, y=True)
        self.plot_roll = self.pw_roll.plot()
        self.plot_roll_cmd = self.pw_roll.plot()
        self.pw_pitch = pg.PlotWidget(title='pitch (deg)')
        self.pw_pitch.setYRange(max=25, min=-25)
        self.pw_pitch.showGrid(x=True, y=True)
        self.plot_pitch = self.pw_pitch.plot()
        self.plot_pitch_cmd = self.pw_pitch.plot()
        self.pw_yaw = pg.PlotWidget(title='yaw (deg)')
        self.pw_yaw.showGrid(x=True, y=True)
        self.plot_yaw = self.pw_yaw.plot()
        self.plot_yaw_cmd = self.pw_yaw.plot()
        layout.addWidget(self.pw_roll)
        layout.addWidget(self.pw_pitch)
        layout.addWidget(self.pw_yaw)
        plot_gb.setLayout(layout)
        return plot_gb

    def attitude_plot_cb(self, step, attitude, attitude_cmd):
        self.update_value_list(self.pitch_list, attitude[0])
        self.update_value_list(self.roll_list, attitude[1])
        self.update_value_list(self.yaw_list, attitude[2])
        if attitude_cmd is not None and len(attitude_cmd) == 3:
             pass
        self.plot_pitch.setData(self.pitch_list, pen=self.pen_red)
        self.plot_roll.setData(self.roll_list, pen=self.pen_red)
        self.plot_yaw.setData(self.yaw_list, pen=self.pen_red)

    def create_reward_plot_groupbox(self):
        reward_plot_groupbox = QGroupBox(title='Reward')
        layout = QHBoxLayout()
        reward_plot_groupbox.setFixedWidth(600)
        self.reward_list = np.full(self.max_len, np.nan)
        self.total_reward_list = np.full(self.max_len, np.nan)
        self.rw_pw_1 = pg.PlotWidget(title='reward')
        self.rw_pw_1.showGrid(x=True, y=True)
        self.rw_p_1 = self.rw_pw_1.plot()
        self.rw_pw_2 = pg.PlotWidget(title='total reward')
        self.rw_pw_2.showGrid(x=True, y=True)
        self.rw_p_2 = self.rw_pw_2.plot()
        layout.addWidget(self.rw_pw_1)
        layout.addWidget(self.rw_pw_2)
        reward_plot_groupbox.setLayout(layout)
        return reward_plot_groupbox

    def reward_plot_cb(self, step, reward, total_reward):
        self.update_value_list(self.reward_list, reward)
        self.update_value_list(self.total_reward_list, total_reward)
        self.rw_p_1.setData(self.reward_list, pen=self.pen_red)
        self.rw_p_2.setData(self.total_reward_list, pen=self.pen_red)

    def create_lgmd_plot_groupbox(self):
        lgmd_plot_groupbox = QGroupBox(title='Lgmd info')
        layout = QHBoxLayout()
        lgmd_plot_groupbox.setFixedWidth(600)
        self.min_dist_to_obs_list = np.full(self.max_len, np.nan)
        self.lgmd_out_list = np.full(self.max_len, np.nan)
        self.lgmd_pw_1 = pg.PlotWidget(title='min distance to obs')
        self.lgmd_pw_1.showGrid(x=True, y=True)
        self.lgmd_p_1 = self.lgmd_pw_1.plot()
        self.lgmd_pw_2 = pg.PlotWidget(title='lgmd output')
        self.lgmd_pw_2.showGrid(x=True, y=True)
        self.lgmd_p_2 = self.lgmd_pw_2.plot()
        self.lgmd_pw_3 = MatplotlibWidget()
        layout.addWidget(self.lgmd_pw_1)
        layout.addWidget(self.lgmd_pw_2)
        lgmd_plot_groupbox.setLayout(layout)
        return lgmd_plot_groupbox

    def lgmd_plot_cb(self, min_dist, lgmd_out, lgmd_split):
        self.update_value_list(self.min_dist_to_obs_list, min_dist)
        self.update_value_list(self.lgmd_out_list, lgmd_out)
        self.lgmd_p_1.setData(self.min_dist_to_obs_list, pen=self.pen_red)
        self.lgmd_p_2.setData(self.lgmd_out_list, pen=self.pen_red)

    def create_traj_plot_groupbox(self):
        traj_plot_groupbox = QGroupBox('Trajectory')
        traj_plot_groupbox.setFixedHeight(600)
        traj_plot_groupbox.setFixedWidth(600)
        layout = QVBoxLayout()
        self.traj_pw = pg.PlotWidget(title='trajectory')
        self.traj_pw.showGrid(x=True, y=True)
        self.traj_pw.setXRange(max=100, min=-100) 
        self.traj_pw.setYRange(max=100, min=-100) 
        self.traj_plot = self.traj_pw.plot()
        self.traj_pw.invertY()
        layout.addWidget(self.traj_pw)
        traj_plot_groupbox.setLayout(layout)
        return traj_plot_groupbox

    def traj_plot_cb(self, uav_trajectory, goal_trajectory):
        self.traj_pw.clear()
        background_list = ['SimpleAvoid', 'NH_center', 'City_400', 'Tree_200', 'Forest']
        if self.cfg.get('options', 'env_name') in background_list and hasattr(self, 'background_img'):
            self.traj_pw.addItem(self.background_img)
        if len(goal_trajectory) > 1:
            goal_x = goal_trajectory[:, 0]
            goal_y = goal_trajectory[:, 1]
            self.traj_pw.plot(goal_x, goal_y, pen=pg.mkPen(color='r', width=3, style=pg.QtCore.Qt.DashLine), name='Goal Path')
            self.traj_pw.plot([goal_x[0]], [goal_y[0]], pen=None, symbol='o', symbolBrush='orange', symbolSize=12)
            self.traj_pw.plot([goal_x[-1]], [goal_y[-1]], pen=None, symbol='x', symbolBrush='red', symbolSize=15)
        elif len(goal_trajectory) == 1:
            goal_x = goal_trajectory[0, 0]
            goal_y = goal_trajectory[0, 1]
            self.traj_pw.plot([goal_x], [goal_y], pen=None, symbol='o', symbolBrush='orange', symbolSize=12)
        if len(uav_trajectory) > 1:
            uav_x = uav_trajectory[:, 0]
            uav_y = uav_trajectory[:, 1]
            self.traj_pw.plot(uav_x, uav_y, pen=pg.mkPen(color='k', width=3), name='UAV Path')
            self.traj_pw.plot([uav_x[0]], [uav_y[0]], pen=None, symbol='o', symbolBrush='green', symbolSize=12)
            self.traj_pw.plot([uav_x[-1]], [uav_y[-1]], pen=None, symbol='star', symbolBrush='blue', symbolSize=15)
        elif len(uav_trajectory) == 1:
            uav_x = uav_trajectory[0, 0]
            uav_y = uav_trajectory[0, 1]
            self.traj_pw.plot([uav_x], [uav_y], pen=None, symbol='star', symbolBrush='blue', symbolSize=15)

def main():
    config_file = 'configs/config_new.ini'
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config_file)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
