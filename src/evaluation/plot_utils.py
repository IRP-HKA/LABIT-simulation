"""
Standalone plotting utilities for force and metrics data.

All functions write files to disk and return the saved path.
They can be used independently of BenchmarkEvaluation.
"""

import os

import matplotlib.pyplot as plt
from evaluation.quality_metric_utils import butter_lowpass_filter
import numpy as np


# ---------------------------------------------------------------------------
# Force plot
# ---------------------------------------------------------------------------

def plot_force_data(
    time_data:  np.ndarray,
    force_data: np.ndarray,
    title:      str  = "Force Data",
    save_path:  str  = None,
) -> str:
    """
    Plot Fx / Fy / Fz over time and save to disk.

    Parameters
    ----------
    time_data  : 1-D array of timestamps [s]
    force_data : (N, 3) array of [Fx, Fy, Fz] forces [N]
    title      : figure title
    save_path  : full file path to save the PNG; if None the figure is shown
                 interactively instead

    Returns
    -------
    save_path if saved, else None
    """
    fig, ax = plt.subplots()
    ax.plot(time_data, force_data[:, 0], label="Force X", color="red")
    ax.plot(time_data, force_data[:, 1], label="Force Y", color="green")
    ax.plot(time_data, force_data[:, 2], label="Force Z", color="blue")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.legend()
    ax.grid()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None


# ---------------------------------------------------------------------------
# Metrics boxplot
# ---------------------------------------------------------------------------

def plot_metrics_boxplot(
    metrics:         dict,
    save_path:       str  = None,
    exclude_subtasks: list = None,
) -> str:
    """
    Create a 4-panel boxplot (E_z, E_xy, S_z, S_xy) for all subtasks.

    Parameters
    ----------
    metrics          : dict keyed by subtask name; each value is a dict with
                       keys "E_z", "E_xy", "S_z", "S_xy" each holding a list
                       of scalar values across trials
    save_path        : full file path (PDF recommended); if None the figure is
                       shown interactively
    exclude_subtasks : list of subtask names to drop before plotting

    Returns
    -------
    save_path if saved, else None
    """
    metrics = {k: v for k, v in sorted(metrics.items())}

    if exclude_subtasks:
        for subtask in exclude_subtasks:
            metrics.pop(subtask, None)

    cm  = 1 / 2.54
    fig, axes = plt.subplots(4, 1, figsize=(21 * cm, 20 * cm), sharex=True)

    for i, metric_name in enumerate(["E_z", "E_xy", "S_z", "S_xy"]):
        ax   = axes[i]
        data = [metrics[subtask][metric_name] for subtask in metrics]
        bplot = ax.boxplot(list(np.log10(data)), labels=metrics.keys(),
                           patch_artist=True)

        for patch, median in zip(bplot['boxes'], bplot['medians']):
            patch.set_facecolor("#c9daf8")
            median.set_color("black")

        if metric_name in ["E_z", "E_xy"]:
            ax.set_yticks(np.arange(-2, 4))
            ax.set_yticklabels(
                ["10$^{-2}$", "10$^{-1}$", "10$^{0}$",
                 "10$^{1}$", "10$^{2}$", "10$^{3}$"]
            )
            unit = "N$^2$"
        else:
            ax.set_yticks(np.arange(-3, 1))
            ax.set_yticklabels(
                ["10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "10$^{0}$"]
            )
            unit = "N"

        ax.set_ylabel(
            "$" + metric_name[:2] + "{" + metric_name[2:] + "}$ [" + unit + "]",
            fontsize=8,
        )
        ax.grid(alpha=0.3)

    axes[-1].set_xticks(np.arange(1, len(metrics) + 1))
    axes[-1].set_xticklabels(metrics.keys(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Boxplot saved to {save_path}")
        return save_path
    else:
        plt.show()
        return None

if __name__ == "__main__":
    # Example usage of the plotting functions
    data = np.load("workcells/sim_param_estimation/results/2026_05_27_13_40_54_trial_1_rnd/2026_05_27_13_41_01_male_peg/2026_05_27_13_41_01_male_peg_inserting.npz")

    order = 6
    fs = 1/0.001 #1000.0  # sample rate, Hz
    cutoff = 5.0  # desired cutoff frequency, Hz

    force_data = data["eef_fts"]

    Fx = force_data[:,0] - np.mean(force_data[0:100,0])
    Fy = force_data[:,1] - np.mean(force_data[0:100,1])
    Fz = -1 * (force_data[:,2] - np.mean(force_data[0:100,2]))

    Fx = butter_lowpass_filter(force_data[:, 0], cutoff, fs, order)
    Fy = butter_lowpass_filter(force_data[:, 1], cutoff, fs, order)
    Fz = butter_lowpass_filter(Fz, cutoff, fs, order)

    
    plot_force_data(data["timestamp"], np.stack([Fx, Fy, Fz], axis=1), title="Trial")

    # example_metrics = {
    #     "subtask_1": {"E_z": np.random.rand(10) * 100, "E_xy": np.random.rand(10) * 100,
    #                   "S_z": np.random.rand(10), "S_xy": np.random.rand(10)},
    #     "subtask_2": {"E_z": np.random.rand(10) * 100, "E_xy": np.random.rand(10) * 100,
    #                   "S_z": np.random.rand(10), "S_xy": np.random.rand(10)},
    #     "subtask_3": {"E_z": np.random.rand(10) * 100, "E_xy": np.random.rand(10) * 100,
    #                   "S_z": np.random.rand(10), "S_xy": np.random.rand(10)},
    # }

    # plot_metrics_boxplot(example_metrics, title="Example Metrics Boxplot")