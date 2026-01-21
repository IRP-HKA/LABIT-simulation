import numpy as np
import os
import matplotlib.pyplot as plt
from qbit.sim_envs.mujoco_env_base import MujocoEnvBase
from datetime import datetime


class DataRecording():
    def __init__(self, task_env_config_path, robot=None, sim_timestep=0.001, live_plotting=False):
        self.init()

        self.robot = robot
        self.sim_timestep = sim_timestep
        self.live_plotting = live_plotting
        
        self.config = MujocoEnvBase.parse_qbit_config_yaml(task_env_config_path)

        self.RESULT_DIR = os.path.join("/workspace/examples/experiment_results/", self.config["data_recording"]["save_folder"])

        if live_plotting:
            plt.ion()
            fig, self.ax = plt.subplots()
            (self.line,) = self.ax.plot([], [], lw=2)
            self.ax.set_xlabel("Timestep")
            self.ax.set_ylabel("Sensor Value")
            self.ax.set_title("Live sensor plot")

        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

    def init(self):
        self.i = 0
        self.timestamp = []
        self.eef_fts = []
        self.eef_pos = []
        self.eef_qua = []
        self.joint_states = []
        self.labels = []
    
    def record(self, label: str):
        # get states
        timestamp = self.sim_timestep * self.i

        current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()
        current_joint_state = self.robot.get_current_joint_state()
        eef_fts = self.robot.get_fts_data(transform_to_base=True)
        
        self.timestamp.append(timestamp)
        self.eef_fts.append(eef_fts)
        self.eef_pos.append(current_eef_pose_T.translation)
        self.eef_qua.append(current_eef_pose_T.quaternion)
        self.joint_states.append(current_joint_state)
        self.labels.append(label)

        if self.live_plotting:
            self.line.set_xdata(np.arange(len(np.array(self.eef_fts)[:,0])))
            self.line.set_ydata(np.array(self.eef_fts)[:,0])
            self.ax.relim()
            self.ax.autoscale_view()
            plt.pause(0.00001)

        self.i += 1

    def save(self,):
        self.filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.savepath = os.path.join(self.RESULT_DIR, self.filename)

        self.timestamp = np.array(self.timestamp)
        self.eef_fts = np.array(self.eef_fts)
        self.eef_pos = np.array(self.eef_pos)
        self.eef_qua = np.array(self.eef_qua)
        self.joint_states = np.array(self.joint_states)
        self.labels = np.array(self.labels)

        np.savez(self.savepath + ".npz",
                 timestamp = self.timestamp,
                 eef_fts = self.eef_fts,
                 eef_pos = self.eef_pos,
                 eef_qua = self.eef_qua,
                 joint_states = self.joint_states,
                 labels = self.labels)
    
        self.print_info()
    
    def print_info(self):
        print("recorded data saved to " + self.savepath)

        print("timestamp: " + str(self.timestamp.shape))
        print("eef_fts: " + str(self.eef_fts.shape))
        print("eef_pos: " + str(self.eef_pos.shape))
        print("eef_qua: " + str(self.eef_qua.shape))
        print("joint_states: " + str(self.joint_states.shape))

        return

    def _plot_label_segments(self, time, values, labels, ax):
        """Plot values over time, color-coded by labels."""
        unique_labels = list(set(labels))
        cmap = plt.get_cmap("tab10")

        for idx, lab in enumerate(unique_labels):
            mask = np.array(labels) == lab
            ax.plot(time[mask], values[mask], '.', 
                    color=cmap(idx), label=lab, markersize=4)

        ax.legend()

    def plot_data(self):
        # load previously saved .npz file
        data = np.load(self.savepath + ".npz")

        # map saved arrays to names used by the plotting code
        time = np.asarray(data["timestamp"])
        force = np.asarray(data["eef_fts"])
        position = np.asarray(data["eef_pos"])
        labels = np.array(data["labels"])

        # ensure arrays are 2D where expected
        if force.ndim == 1:
            force = force.reshape(-1, 1)
        if position.ndim == 1:
            position = position.reshape(-1, 1)

        fig, axs = plt.subplots(2, 3, figsize=(12,6))

        for idx in range(3):
            ax = axs[0, idx]
            self._plot_label_segments(time, force[:, idx], labels, ax)
            ax.set_title(["Fx","Fy","Fz"][idx])
            ax.set_ylim((-50,50))
            ax.set_xlabel("t [s]")
            ax.set_ylabel("Force [N]")

            ax = axs[1, idx]
            self._plot_label_segments(time, position[:, idx], labels, ax)
            ax.set_title(["x","y","z"][idx])
            ax.set_xlabel("t [s]")
            ax.set_ylabel("Pos [m]")

        plt.tight_layout()
        plt.savefig(self.savepath + ".png")
        plt.close()
        plt.cla()
        plt.clf()

        return
