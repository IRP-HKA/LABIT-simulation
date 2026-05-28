import os
import sys
sys.path.append(os.getcwd())
import glob
import numpy as np

from src.evaluation.quality_metric_utils import metric_signal_energy, metric_signal_smoothness, butter_lowpass_filter
from src.evaluation.plot_utils import plot_force_data, plot_metrics_boxplot


class BenchmarkEvaluation:
    def __init__(self, benchmark_data_path, metrics_file=None):
        self.benchmark_data_path = benchmark_data_path
        self.metrics_file = metrics_file or os.path.join(benchmark_data_path, "qbit_metrics.npy")
        self.metrics = {}
        
        # Filter parameters
        self.order = 6
        self.fs = 1/0.0005 #1000.0  # sample rate, Hz
        self.cutoff = 5.0  # desired cutoff frequency, Hz
    
    def process_trial_folders(self):
        """Find and process all trial folders in the benchmark directory."""
        trial_folders = [
            os.path.join(self.benchmark_data_path, folder)
            for folder in os.listdir(self.benchmark_data_path)
            if "trial" in folder.lower() and os.path.isdir(os.path.join(self.benchmark_data_path, folder))
        ]

        for trial_folder in trial_folders:
            self._process_trial(trial_folder)
        
        self._save_metrics()
    
    def _process_trial(self, trial_folder):
        """Process all subtasks in a single trial folder."""
        subtask_folders = [
            os.path.join(trial_folder, subfolder)
            for subfolder in os.listdir(trial_folder)
            if os.path.isdir(os.path.join(trial_folder, subfolder))
        ]

        for subtask_folder in subtask_folders:
            self._process_subtask(subtask_folder)
    
    def _process_subtask(self, subtask_folder):
        """Process a single subtask folder containing .npz files."""
        subtask_name = os.path.basename(subtask_folder)
        subtask_name = subtask_name[20:] # removing timestamp from name
        subtask_name = name_mapping[subtask_name]

        # Find all .npz files in the subtask folder
        npz_files = sorted(glob.glob(os.path.join(subtask_folder, "*.npz")))
        
        if not npz_files:
            print(f"No .npz files found in {subtask_folder}")
            return
        
        force_data = []
        time_data = []
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            forces = data["eef_fts"]
            times = data["timestamp"]
            print(f"Processing: {npz_file}")
            
            force_data.append(forces)
            time_data.append(times)
        
        # Concatenate all force and time data
        force_data = np.concatenate(force_data)
        time_data = np.concatenate(time_data)
        
        # Compute metrics and plot
        self._compute_and_plot_metrics(subtask_name, subtask_folder, time_data, force_data)
    
    def _compute_and_plot_metrics(self, subtask_name, subtask_folder, time_data, force_data):
        """Compute metrics and create plots for force data."""
        # Apply low-pass filter
        Fz = -1 * (force_data[:,2] - np.mean(force_data[0:100,2]))
        Fx = butter_lowpass_filter(force_data[:, 0], self.cutoff, self.fs, self.order)
        Fy = butter_lowpass_filter(force_data[:, 1], self.cutoff, self.fs, self.order)
        Fz = butter_lowpass_filter(Fz, self.cutoff, self.fs, self.order)
        
        # Compute force in plane perpendicular to Fz
        Fxy = np.stack([Fx, Fy], axis=1)
        Fxy = np.linalg.norm(Fxy, axis=1)
        
        # Calculate metrics
        E_z = metric_signal_energy(F=Fz)
        E_xy = metric_signal_energy(F=Fxy)
        S_z = metric_signal_smoothness(F=Fz)
        S_xy = metric_signal_smoothness(F=Fxy)
        
        # Store metrics
        if subtask_name not in self.metrics:
            self.metrics[subtask_name] = {"E_z": [], "E_xy": [], "S_z": [], "S_xy": []}
        
        self.metrics[subtask_name]["E_z"].append(E_z)
        self.metrics[subtask_name]["E_xy"].append(E_xy)
        self.metrics[subtask_name]["S_z"].append(S_z)
        self.metrics[subtask_name]["S_xy"].append(S_xy)
        
        # Plot force data
        self._plot_force_data(subtask_name, subtask_folder, time_data, np.stack([Fx, Fy, Fz], axis=1))
    
    def _plot_force_data(self, subtask_name, subtask_folder, time_data, force_data):
        """Create and save force plot."""
        plot_path = os.path.join(subtask_folder, f"{subtask_name}_force_plot.png")
        plot_force_data(
            time_data=time_data,
            force_data=force_data,
            title=f"Force Data for Subtask: {subtask_name}",
            save_path=plot_path,
        )
    
    def _save_metrics(self):
        """Save metrics to numpy file."""
        np.save(self.metrics_file, self.metrics)
        print(f"Metrics saved to {self.metrics_file}")
    
    def plot_metrics(self, exclude_subtasks=None):
        """Create boxplots for all metrics."""
        if not os.path.exists(self.metrics_file):
            print(f"Metrics file not found: {self.metrics_file}")
            return

        metrics = np.load(self.metrics_file, allow_pickle=True).item()
        boxplot_path = os.path.join(self.benchmark_data_path, "metrics_boxplot_all.pdf")
        plot_metrics_boxplot(
            metrics=metrics,
            save_path=boxplot_path,
            exclude_subtasks=exclude_subtasks,
        )


if __name__ == "__main__":
    benchmark_data_path = r"workcells/sim_param_estimation/results" #r"workcells/labit_benchmark/results"
    name_mapping = {"pcb": "PCB", 
                    "plug_inside_loose_1": "Plug_inside_loose_1",
                    "plug_inside_loose_2": "Plug_inside_loose_2",
                    "plug_outside_loose": "Plug_oustide_loose",
                    "positioning_pin_d5_20_1": "Positioning_Pin_4",
                    "positioning_pin_d5_20_2": "Positioning_Pin_3",
                    "positioning_pin_d5_20_3": "Positioning_Pin_5",
                    "positioning_pin_d5_20_4": "Positioning_Pin_6",
                    "positioning_pin_d5_20_5": "Positioning_Pin_1",
                    "positioning_pin_d5_20_6": "Positioning_Pin_2",
                    "bolt_rotor": "Rotor",
                    "housing_middle_grasp_target": "Housing_Middle",
                    "gearwheel_teeth_35_mod_2_1": "Gearwheel_Teeth_1",
                    "gearwheel_teeth_35_mod_2_2": "Gearwheel_Teeth_2",
                    "tube_nozzle": "Housing_Top",
                    "screw_m5_16_hexagon_head_1": "Screw_M5x16_4",
                    "screw_m5_16_hexagon_head_2": "Screw_M5x16_3",
                    "screw_m5_16_hexagon_head_3": "Screw_M5x16_5",
                    "screw_m5_16_hexagon_head_4": "Screw_M5x16_2",
                    "screw_m5_16_hexagon_head_5": "Screw_M5x16_1",
                    "o_ring_grasp_target": "O-Ring",
                    "cover_plate": "Coverplatte",
                    "tube": "Tube",
                    "tube_clamp": "Tube_Clamp",
                    "housing_assembly_grasp_target": "Module"}

    # Initialize evaluator
    evaluator = BenchmarkEvaluation(benchmark_data_path)
    
    # Process all trial folders and compute metrics
    evaluator.process_trial_folders()
    
    # Plot metrics (optionally exclude certain subtasks)
    evaluator.plot_metrics() # ["screw_m5_16_hexagon_head_4", "screw_m5_16_hexagon_head_5"]