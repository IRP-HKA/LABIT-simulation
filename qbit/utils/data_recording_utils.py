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
        
        self.subtask_name = "default"
        self.primitive_name = "default"
        self.i = 0

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
        self.timestamp = []
        self.eef_fts = []
        self.eef_pos = []
        self.eef_qua = []
        self.joint_states = []

    def set_subtask_name(self, name):
        self.subtask_name = name.replace("_body","")
        self.subtask_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def set_primitive_name(self, name):
        self.primitive_name = name
        self.primitive_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    def record(self):
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

        if self.live_plotting:
            self.line.set_xdata(np.arange(len(np.array(self.eef_fts)[:,0])))
            self.line.set_ydata(np.array(self.eef_fts)[:,0])
            self.ax.relim()
            self.ax.autoscale_view()
            plt.pause(0.00001)

        self.i += 1

    def save(self):
        subtask_dir = os.path.join(self.RESULT_DIR, f"{self.subtask_timestamp}_{self.subtask_name}")
        if not os.path.exists(subtask_dir):
            os.makedirs(subtask_dir)
        
        # Save file as: timestamp_subtaskname_primitivename
        self.savepath = os.path.join(subtask_dir, f"{self.primitive_timestamp}_{self.subtask_name}_{self.primitive_name}")

        self.timestamp = np.array(self.timestamp)
        self.eef_fts = np.array(self.eef_fts)
        self.eef_pos = np.array(self.eef_pos)
        self.eef_qua = np.array(self.eef_qua)
        self.joint_states = np.array(self.joint_states)
        
        np.savez(self.savepath + ".npz",
                 timestamp=self.timestamp,
                 eef_fts=self.eef_fts,
                 eef_pos=self.eef_pos,
                 eef_qua=self.eef_qua,
                 joint_states=self.joint_states)
    
        self.print_info()
        self.init()

    
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

    def plot_primitive(self, filepath=None):
        """Plot force/torque and position data from a primitive .npz file."""
        if filepath is None:
            filepath = self.savepath
        
        data = np.load(filepath + ".npz")
        
        time = np.asarray(data["timestamp"])
        force = np.asarray(data["eef_fts"])
        position = np.asarray(data["eef_pos"])
        
        if force.ndim == 1:
            force = force.reshape(-1, 1)
        if position.ndim == 1:
            position = position.reshape(-1, 1)
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        
        for idx in range(3):
            ax = axs[0, idx]
            ax.plot(time, force[:, idx], 'b-', linewidth=1)
            ax.set_title(["Fx", "Fy", "Fz"][idx])
            ax.set_xlabel("t [s]")
            ax.set_ylabel("Force [N]")
            ax.grid(True, alpha=0.3)
            
            ax = axs[1, idx]
            ax.plot(time, position[:, idx], 'r-', linewidth=1)
            ax.set_title(["x", "y", "z"][idx])
            ax.set_xlabel("t [s]")
            ax.set_ylabel("Pos [m]")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath + ".png")
        plt.close()

    def plot_multiple_primitives(self, filepaths, figsize=(12, 6)):
        """Plot force/torque and position data from multiple primitive .npz files."""
        fig, axs = plt.subplots(2, 3, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(filepaths)))
        
        for file_idx, filepath in enumerate(filepaths):
            data = np.load(filepath + ".npz")
            
            time = np.asarray(data["timestamp"])
            force = np.asarray(data["eef_fts"])
            position = np.asarray(data["eef_pos"])
            
            if force.ndim == 1:
                force = force.reshape(-1, 1)
            if position.ndim == 1:
                position = position.reshape(-1, 1)
            
            for idx in range(3):
                axs[0, idx].plot(time, force[:, idx], color=colors[file_idx], 
                                linewidth=1, label=f"Primitive {file_idx}")
                axs[0, idx].set_title(["Fx", "Fy", "Fz"][idx])
                axs[0, idx].set_xlabel("t [s]")
                axs[0, idx].set_ylabel("Force [N]")
                axs[0, idx].grid(True, alpha=0.3)
                
                axs[1, idx].plot(time, position[:, idx], color=colors[file_idx], 
                                linewidth=1, label=f"Primitive {file_idx}")
                axs[1, idx].set_title(["x", "y", "z"][idx])
                axs[1, idx].set_xlabel("t [s]")
                axs[1, idx].set_ylabel("Pos [m]")
                axs[1, idx].grid(True, alpha=0.3)
        
        axs[0, 0].legend()
        plt.tight_layout()
        plt.savefig("multiple_primitives_plot.png")
        plt.close()
        return fig, axs

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



if __name__ == "__main__":
    mDataRecorder = DataRecording(task_env_config_path="qbit/configs/envs/ur5e_labit_benchmark.yaml")
    # mDataRecorder.plot_primitive(filepath="examples/experiment_results/trial_0/2026_01_21_15_03_08_positioning_pin_d5_20_2/2026_01_21_15_39_03_positioning_pin_d5_20_2_grasping")


    mDataRecorder.plot_multiple_primitives(filepaths=[
        "examples/experiment_results/trial_0/2026_01_21_15_03_08_positioning_pin_d5_20_2/2026_01_21_15_03_08_positioning_pin_d5_20_2_grasping",
        "examples/experiment_results/trial_0/2026_01_21_15_03_08_positioning_pin_d5_20_2/2026_01_21_15_16_06_positioning_pin_d5_20_2_grasping",])