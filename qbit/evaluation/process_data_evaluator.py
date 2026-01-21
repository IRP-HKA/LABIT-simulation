"""
Data Processing Tool for recording and evaluating the process data
"""


import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg') # using non-interactive backend
import matplotlib.pyplot as plt

from qbit.evaluation.quality_metric_utils import metric_signal_energy, metric_signal_smoothness, butter_lowpass_filter
from qbit.evaluation.plot_radar_chart import radar_factory


DT = 0.001

class ProcessDataEvaluator:

    def __init__(self,
                 data_file_path: str,
                 file_name_prefix: str = "record"):
        self._data_path = data_file_path
        self._file_name_prefix = file_name_prefix
        
        if not os.path.isdir(self._data_path):
            os.makedirs(self._data_path)
            print(f"Create a new directory for data storage: {self._data_path}")

        self.record_id = 0
        self.clear_buffer()

    def clear_buffer(self):
        self._data = {
            "t": [],
            "eef_fts": [],
            "eef_pos": [],
            "eef_qua": [],
            "joint_states": [],
            "n_contacts": []
        }
        return
    
    def record(self,
               timestamp: np.ndarray,
               eef_fts: np.ndarray,
               eef_pos: np.ndarray,
               eef_qua: np.ndarray,
               joint_states: np.ndarray,
               n_contacts: int = 0,
               ):
        self._data["t"].append(timestamp)
        self._data["eef_fts"].append(eef_fts)
        self._data["eef_pos"].append(eef_pos)
        self._data["eef_qua"].append(eef_qua)
        self._data["joint_states"].append(joint_states)
        self._data["n_contacts"].append(n_contacts)
        return

    def save(self):
        f_path = f"{self._data_path}/{self._file_name_prefix}_{self.record_id}.npy"
        np.save(f_path,
                self._data,
                allow_pickle=True
                )
        print(f"Data saved to {self._data_path}/{f_path}")
        
        self.record_id += 1
        return
    
    def read(self):
        self._data = np.load(f"{self._data_path}/{self._file_name}",
                       allow_pickle=True).item()
        # print(self._data["t"])
        return
    
    def get_quality_metrics_batch(self):
        order = 6
        fs = 1000.0         # sample rate, Hz
        cutoff = 5.0        # desired cutoff frequency of the filter, Hz
        
        E_z, E_xy, S_z, S_xy, completion_time = [], [], [], [], []
        
        f_paths = sorted(glob.glob(os.path.join(self._data_path, "*.npy")))
        
        for f_path in f_paths:
            
            data = np.load(f_path, allow_pickle=True).item()
            
            print(np.array(data['eef_fts']).shape)

            wrech = np.array(data['eef_fts'])
            F_x = wrech[:, 0]
            F_y = wrech[:, 1]
            F_z = wrech[:, 2]
            F_x = butter_lowpass_filter(F_x, cutoff, fs, order)
            F_y = butter_lowpass_filter(F_y, cutoff, fs, order)
            F_z = butter_lowpass_filter(F_z, cutoff, fs, order)

            F_xy = np.stack([F_x, F_y], axis=1)
            F_xy = np.linalg.norm(F_xy, axis=1)
            
            print("F_z: ", F_z)
            
            E_z.append(metric_signal_energy(F=F_z))
            S_z.append(metric_signal_smoothness(F=F_z))

            E_xy.append(metric_signal_energy(F=F_xy))
            S_xy.append(metric_signal_smoothness(F=F_xy))

            completion_time.append(data['t'][-1])

        return completion_time, E_z, E_xy, S_z, S_xy

    @staticmethod
    def plot_radar(data, store_to_file):
        theta = radar_factory(5, frame='polygon')
        fig, ax = plt.subplots(nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
        
        spoke_labels = ["T", "$E_z$", "$E_{xy}$", "$S_{xy}$", "$S_z$"]
        ax.plot(theta, data, linewidth=2.0, color="#9AC7BF")
        ax.fill(theta, data, facecolor="#9AC7BF", alpha=0.25, label='_nolegend_')

        ax.set_varlabels(spoke_labels)
        ax.set_yticklabels([])
        ax.tick_params(axis='x', which='major', labelsize=20)

        plt.tight_layout()
        
        png_path = store_to_file
        print(f"Exporting the position and velocity plot to {png_path}")
        plt.savefig(png_path)
        return
        
    
    def get_radar_plot(self):
        
        # Compute the relative metrics
        data = self.get_quality_metrics_batch()
        metrics_max = np.array(data).max(axis=1)
        metrics_min = np.array(data).min(axis=1)
        # metrics_min[0] = 1.0
        print("Matrics max: ", metrics_max)
        print("Matrics min: ", metrics_min)
        
        print("Mean: ", np.mean(data, axis=1))
        
        metric_normalized = (np.mean(data, axis=1) - metrics_max) / (metrics_min - metrics_max)
        png_path = os.path.splitext(f"{self._data_path}/{self._file_name_prefix}")[0] + "_radar_plot.png"
        # Plot the radar chart
        self.plot_radar(metric_normalized,
                        store_to_file=png_path)



    def _get_t_index(self,
                     t,
                     t_min = None,
                     t_max = None):
        if t_min is None:
            t_min = 0
        if t_max is None:
            t_max = t[-1]
        if t_max > t[-1]:
            t_max = t[-1]
        if t_min > t_max:
            t_min = 0
            print(f"t_min is greater than the recorded timestamp, set t_min to 0")

        t_idx = np.where((t >= t_min) & (t <= t_max))[0]
        return t_idx
    
    def find_t_range_by_force(self, force_z_threshold = 0.01):
        t = np.array(self._data['t'])
        force_z = np.array(self._data['eef_fts'])[:, 2]
        indices = np.where(np.abs(force_z) > force_z_threshold)[0]
        if len(indices) == 0:
            print("No force values greater than 0.01 found.")
            return None, None
        t_min = t[indices[0]]
        t_max = t[indices[-1]]
        print(f"Force detected between {t_min} and {t_max}")
        return t_min, t_max
    
    def plot_force(self,
                   t_min = None,
                   t_max = None,
                   file_name_postfix = '_force.png'):

        t = np.array(self._data['t'])
        data = np.array(self._data['eef_fts']).T
        print(data.shape)
        force_x = data[0, :]
        force_y = data[1, :]
        force_z = data[2, :]
        
        t_idx = self._get_t_index(t, t_min, t_max)
        
        plt.figure(figsize=(50, 6))
        plt.plot(t[t_idx], force_x[t_idx], label="Force X", lw=2)
        plt.plot(t[t_idx], force_y[t_idx], label="Force Y", lw=2)
        plt.plot(t[t_idx], force_z[t_idx], label="Force Z", lw=2)

        # plt.ylim(-20, 20)
        
        plt.xlabel("Time (s)")
        plt.ylabel("Force")
        plt.title("Force Sensor Data Over Time")
        plt.legend()
        plt.grid(True)
        
        png_path = os.path.splitext(f"{self._data_path}/{self._file_name_prefix}_{self.record_id}")[0] + file_name_postfix
        print(f"Exporting the force plot to {png_path}")
        plt.savefig(png_path)
        
    def plot_n_contacts(self,
                        file_name_postfix = '_n_contacts.png'):

        t = np.array(self._data['t'])
        n_contacts = np.array(self._data['n_contacts'])

        t_idx = self._get_t_index(t)

        plt.figure(figsize=(50, 6))
        plt.plot(t[t_idx], n_contacts[t_idx], label="Number of Contacts", lw=2)

        plt.xlabel("Time (s)")
        plt.ylabel("Number of Contacts")
        plt.title("Number of Contacts Over Time")
        plt.legend()
        plt.grid(True)

        png_path = os.path.splitext(f"{self._data_path}/{self._file_name_prefix}_{self.record_id}")[0] + file_name_postfix
        print(f"Exporting the number of contacts plot to {png_path}")
        plt.savefig(png_path)
        return
        

    def plot_force_entire(self):
        self.plot_force(file_name_postfix='_force_entire.png')
    
    def plot_position(self,
                      t_min,
                      t_max):
        t = np.array(self._data['t'])
        data = np.array(self._data['eef_pos']).T
        print(data.shape)
        pos_x = data[0, :]
        pos_y = data[1, :]
        pos_z = data[2, :]
        t_idx = self._get_t_index(t, t_min, t_max)

        # Calculate velocity in z axis
        velocity_z = np.gradient(pos_z, t)

        fig, axs = plt.subplots(4, 1, figsize=(50, 24))

        axs[0].plot(t[t_idx], pos_x[t_idx], label="Position X", lw=2)
        axs[0].set_ylim(pos_x[t_idx].min(), pos_x[t_idx].max())
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position X")
        axs[0].set_title("End Effector Position X Over Time")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(t[t_idx], pos_y[t_idx], label="Position Y", lw=2)
        axs[1].set_ylim(pos_y[t_idx].min(), pos_y[t_idx].max())
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Position Y")
        axs[1].set_title("End Effector Position Y Over Time")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(t[t_idx], pos_z[t_idx], label="Position Z", lw=2)
        axs[2].set_ylim(pos_z[t_idx].min(), pos_z[t_idx].max())
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Position Z")
        axs[2].set_title("End Effector Position Z Over Time")
        axs[2].legend()
        axs[2].grid(True)

        axs[3].plot(t[t_idx], velocity_z[t_idx], label="Velocity Z", lw=2)
        axs[3].set_ylim(velocity_z[t_idx].min(), velocity_z[t_idx].max())
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Velocity Z")
        axs[3].set_title("End Effector Velocity Z Over Time")
        axs[3].legend()
        axs[3].grid(True)

        fig.tight_layout()
        png_path = os.path.splitext(f"{self._data_path}/{self._file_name_prefix}_{self.record_id}")[0] + "_position_velocity.png"
        print(f"Exporting the position and velocity plot to {png_path}")
        plt.savefig(png_path)
        return
    