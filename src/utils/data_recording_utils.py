from glob import glob
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import imageio

from datetime import datetime

from src.sim_envs.mujoco_env_base import MujocoEnvBase
from src.evaluation.quality_metric_utils import butter_lowpass_filter

class DataRecording():
    def __init__(self, task_env_config_path, robot=None, sim_timestep=0.001, live_plotting=False, cameras=None, fps=24):
        self._cameras = cameras or {}
        self._fps = fps
        self.init()

        self.robot = robot
        self.sim_timestep = sim_timestep
        self.live_plotting = live_plotting
        
        self.subtask_name = "default"
        self.primitive_name = "default"
        self.i = 0

        self.config = MujocoEnvBase.parse_qbit_config_yaml(task_env_config_path)

        self.RESULT_DIR = os.path.join(os.path.dirname(sys.argv[0]), "results", datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")+self.config["data_recording"]["save_folder"])

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
        self.joint_positions = []
        self.joint_velocities = []
        self._frames = {name: [] for name in self._cameras}

    def record_frame(self, mj_data):
        """Capture one frame from each registered camera and append to its buffer."""
        for name, (renderer, camera) in self._cameras.items():
            renderer.update_scene(mj_data, camera=camera)
            self._frames[name].append(renderer.render().copy())

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
        self.joint_positions.append(current_joint_state[0])
        self.joint_velocities.append(current_joint_state[1])

        if self.live_plotting:
            self.line.set_xdata(np.arange(len(np.array(self.eef_fts)[:,0])))
            self.line.set_ydata(np.array(self.joint_positions)[:,2])
            self.ax.relim()
            self.ax.autoscale_view()
            plt.pause(0.00001)

        self.i += 1

    def save(self):
        self.subtask_dir = os.path.join(self.RESULT_DIR, f"{self.subtask_timestamp}_{self.subtask_name}")
        if not os.path.exists(self.subtask_dir):
            os.makedirs(self.subtask_dir)
        
        # Save file as: timestamp_subtaskname_primitivename
        self.savepath = os.path.join(self.subtask_dir, f"{self.primitive_timestamp}_{self.subtask_name}_{self.primitive_name}")

        self.timestamp = np.array(self.timestamp)
        self.eef_fts = np.array(self.eef_fts)
        self.eef_pos = np.array(self.eef_pos)
        self.eef_qua = np.array(self.eef_qua)
        self.joint_positions = np.array(self.joint_positions)
        self.joint_velocities = np.array(self.joint_velocities)
        
        np.savez(self.savepath + ".npz",
                 timestamp=self.timestamp,
                 eef_fts=self.eef_fts,
                 eef_pos=self.eef_pos,
                 eef_qua=self.eef_qua,
                 joint_positions=self.joint_positions,
                 joint_velocities=self.joint_velocities,
                 )

        for name, frames in self._frames.items():
            if frames:
                imageio.mimsave(self.savepath + f"_{name}.mp4", frames, fps=self._fps)

        self.print_info()
        self.cleanup()
        self.init()

    def cleanup(self):
        del self.timestamp
        del self.eef_fts
        del self.eef_pos
        del self.eef_qua
        del self.joint_positions
        del self.joint_velocities
        del self._frames
        
    def print_info(self):
        print("recorded data saved to " + self.savepath)

        print("timestamp: " + str(self.timestamp.shape))
        print("eef_fts: " + str(self.eef_fts.shape))
        print("eef_pos: " + str(self.eef_pos.shape))
        print("eef_qua: " + str(self.eef_qua.shape))
        print("joint_positions: " + str(self.joint_positions.shape))
        print("joint_velocities: " + str(self.joint_velocities.shape))

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
        fig, axs = plt.subplots(3, 6, figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(filepaths)))
        unique_labels = []
        
        for file_idx, filepath in enumerate(filepaths):
            data = np.load(filepath)
            
            time = np.asarray(data["timestamp"])
            force = np.asarray(data["eef_fts"])
            position = np.asarray(data["eef_pos"])
            
            fs = 1/self.sim_timestep
            cutoff = 33.0
            force[:,0] = butter_lowpass_filter(force[:,0], cutoff=cutoff, fs=fs, order=4)
            force[:,1] = butter_lowpass_filter(force[:,1], cutoff=cutoff, fs=fs, order=4)
            # force[:,2] -= force[0,2]
            force[:,2] = butter_lowpass_filter(force[:,2], cutoff=cutoff, fs=fs, order=4)

            # joint_states = np.asarray(data["joint_states"])
            joint_positions = np.asarray(data["joint_positions"])
            joint_velocities = np.asarray(data["joint_velocities"])
            
            primitive_name = os.path.basename(filepath).replace(".npz", "").split("_")[-1]
            if primitive_name not in unique_labels:
                unique_labels.append(primitive_name)
                label = primitive_name
            else:
                label = "_nolegend_"

            if force.ndim == 1:
                force = force.reshape(-1, 1)
            if position.ndim == 1:
                position = position.reshape(-1, 1)
            
            color_idx = unique_labels.index(primitive_name)                
            for idx in range(3):
                axs[0, idx].plot(time, force[:, idx], color=colors[color_idx], 
                                linewidth=1, label=label)
                axs[0, idx].set_title(["Fx", "Fy", "Fz"][idx])
                axs[0, idx].set_xlabel("t [s]")
                axs[0, idx].set_ylabel("Force [N]")
                axs[0, idx].grid(True, alpha=0.3)
                
                axs[1, idx].plot(time, position[:, idx], color=colors[color_idx], 
                                linewidth=1, label=label)
                axs[1, idx].set_title(["x", "y", "z"][idx])
                axs[1, idx].set_xlabel("t [s]")
                axs[1, idx].set_ylabel("Position [m]")
                axs[1, idx].grid(True, alpha=0.3)
            
            for idx in range(6):
                axs[2, idx].plot(time, np.squeeze(joint_positions[:, idx]), "-", color=colors[color_idx], 
                                    linewidth=1, label=label)
                axs[2, idx].set_title(f"Joint {idx + 1}")
                axs[2, idx].set_xlabel("t [s]")
                axs[2, idx].set_ylabel("Position [rad]")
                axs[2, idx].grid(True, alpha=0.3)
        axs[0,3].remove()
        axs[0,4].remove()
        axs[0,5].remove()
        axs[1,3].remove()
        axs[1,4].remove()
        axs[1,5].remove()
        axs[0, 2].legend(loc='upper center', bbox_to_anchor=(0.5, 2))
        plt.tight_layout()
        plt.savefig(os.path.join(*filepath.split(os.sep)[:-1], "multiple_primitives_plot.png"))
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
    mDataRecorder = DataRecording(task_env_config_path="src/configs/envs/ur5e_plug_insertion.yaml")
    # mDataRecorder.plot_primitive(filepath="examples/experiment_results/trial_0/2026_01_21_15_03_08_positioning_pin_d5_20_2/2026_01_21_15_39_03_positioning_pin_d5_20_2_grasping")

    subtask_folder = "workcells/plug_insertion/results/2026_05_12_16_22_32_trial_1_rnd/2026_05_12_16_22_35_dsub25_male"
    filepaths = sorted(glob(os.path.join(subtask_folder, "*.npz")))

    mDataRecorder.plot_multiple_primitives(filepaths=filepaths)