import os
import sys
import signal
import glob
import copy
import yaml
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import trimesh
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from rosbags.rosbag1 import Reader as BagReader
from rosbags.typesys import Stores, get_typestore

from src.utils.tf_utils import T
from src.utils.mj_viewer_utils import update_view_camera_parameter
from src.utils.mujoco_utils import get_relative_pose, convert_quat_to_wxyz
from src.utils.data_recording_utils import DataRecording
from src.sim_envs.mujoco_env_insertion import MujocoEnvBase
from src.objects.mesh_object import MeshObject
from src.objects.decomposed_object import DecomposedObject
from src.objects.sphered_object import SpheredObject


ENV_CONFIG_PATH = "/workspace/src/configs/envs/ur5e_sim_param_estimation.yaml"
REAL_DATA_FOLDER = "/workspace/workcells/sim_param_estimation/data/unprocessed/rectangle_run_01"
BAG_PATH = "/workspace/workcells/sim_param_estimation/data/unprocessed/peg_in_hole/positionctrl_2025-01-17-13-41-03.bag"
NUM_RUNS = 1
SIM_TIMESTEP = 0.01 # Second
POS_RANDOM_LIMIT = 0.000 # meter
_EEF_OFFSET = 0.162 #0.21


class PositionBasedInsertion(MujocoEnvBase):
    # ------------------------------------------------------------------
    # PSO parameter optimisation
    # ------------------------------------------------------------------
    # Peg (task_objects[0]) may use mesh, coacd or shrinking_sphere.
    # Hole (task_objects[1]) may only use coacd or shrinking_sphere.
    _PSO_PEG_MESH_TYPES  = ['coacd', 'shrinking_sphere', 'mesh']
    _PSO_HOLE_MESH_TYPES = ['coacd', 'shrinking_sphere']
    _PSO_NOMINAL_DENSITY = 1190.0  # build-time density used for mass scaling

    # PSO search-space bounds (13D):
    # [peg_mesh_idx, hole_mesh_idx, density, solref×2, solimp×5, friction×3]
    _PSO_BOUNDS_LO = np.array([0.0,   0.0,   500.0,  0.001, 0.1,  0.5,  0.5,  1e-4, 0.1,  1.0,  0.1,  1e-3, 1e-4])
    _PSO_BOUNDS_HI = np.array([2.999, 1.999, 10000.0, 0.1,   2.0,  0.999,0.999,0.01, 0.9,  6.0,  3.0,  0.1,  0.01])
    _PSO_ITERATIONS = 100
    _PSO_PARTICLES = 10

    def __init__(self,
                 task_env_config_path: str,
                 sim_timestep: float = 0.001,
                 rendering_timestep: float = 0.033,
                 rt_factor: float = 0.0,
                 headless: bool = True,
                 server_modus: bool = False,
                 ):

        super(PositionBasedInsertion, self).__init__(
            task_env_config_path,
            sim_timestep,
            rendering_timestep,
            rt_factor,
            headless,
            server_modus,
        )
        
        self._mj_renderer = mujoco.Renderer(self._mj_model, height=720, width=1280)

        self.cam = mujoco.MjvCamera()
        self.cam.azimuth = 0
        self.cam.elevation = -60
        self.cam.distance = 1.0
        self.cam.lookat = [-0.4, 0, 1]

        self.cam_top_view = mujoco.MjvCamera()
        self.cam_top_view.azimuth = 0.0
        self.cam_top_view.elevation = -90.0
        self.cam_top_view.distance = 1.427
        self.cam_top_view.lookat = [-0.268, -0.091, 1.0]

        fps = 24
        self.iterations_per_frame = int(1 / self._sim_timestep / fps)

        self.data_recording = DataRecording(
            task_env_config_path=task_env_config_path,
            robot=self.robot,
            sim_timestep=sim_timestep,
            live_plotting=False,
            cameras={
                "default_view": (self._mj_renderer, self.cam),
                "top_view": (self._mj_renderer, self.cam_top_view),
            },
            fps=fps,
        )

    def simulation_task(self, viewer=None):
        # --- load EEF trajectory from bag ---
        typestore = get_typestore(Stores.ROS1_NOETIC)
        raw_poses = []
        raw_wrenches = []
        with BagReader(BAG_PATH) as reader:
            pose_conns = [c for c in reader.connections if c.topic == '/robot_pose']
            ft_conns   = [c for c in reader.connections if c.topic == '/ftn_axia']
            for conn, ts, rawdata in reader.messages(connections=pose_conns):
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                p = msg.pose.position
                q = msg.pose.orientation
                raw_poses.append([ts * 1e-9, p.x, p.y, p.z + _EEF_OFFSET, q.x, q.y, q.z, q.w])
            for conn, ts, rawdata in reader.messages(connections=ft_conns):
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                f = msg.force; t_ = msg.torque
                raw_wrenches.append([ts * 1e-9, f.x, f.y, f.z, t_.x, t_.y, t_.z])

        raw_poses = np.array(raw_poses)
        real_wrenches = np.array(raw_wrenches)
        real_wrenches[:, 0] -= real_wrenches[0, 0]  # relative time

        pose_times = raw_poses[:, 0] - raw_poses[0, 0]  # relative seconds
        pose_pos = raw_poses[:, 1:4]   # (N,3) EEF position in robot base frame
        pose_quat = raw_poses[:, 4:8]  # (N,4) xyzw quaternion
        slerp = Slerp(pose_times, R.from_quat(pose_quat))
        
        # --- precompute full joint trajectory via IK ---
        n_steps = int(pose_times[-1] / self._sim_timestep)
        joint_traj = np.zeros((n_steps, 6))
        q_prev = self.robot.get_current_joint_state()[0]
        print(f"Precomputing IK for {n_steps} steps ...")
        for i in range(n_steps):
            t = min(i * self._sim_timestep, pose_times[-1])
            p_t = np.array([np.interp(t, pose_times, pose_pos[:, j]) for j in range(3)])
            q_t = slerp(t).as_quat()  # xyzw
            q_prev = self.robot._ik.ik(T(p_t, q_t)._matrix, q_prev)
            joint_traj[i] = q_prev
            if i % 500 == 0:
                print(f"  IK {i}/{n_steps}", end="\r")
        print(f"IK precomputation done.")

        # self.data_recording.set_subtask_name(name="male_peg")
        # self.data_recording.set_primitive_name(name="inserting")

        # self._mj_data.ctrl[0:6] = joint_traj[0]
        # for _ in range(2000):  # hold initial position for a few steps
        #     self.step_mj_simulation()
        #     if viewer is not None:
        #         viewer.sync()

        # # --- replay precomputed joint trajectory ---
        # for i in range(n_steps):
        #     self._mj_data.ctrl[0:6] = joint_traj[i]
        #     self.step_mj_simulation()
        #     self.data_recording.record()
        #     if i % self.iterations_per_frame == 0:
        #         self.data_recording.record_frame(self._mj_data)
        #     if viewer is not None:
        #         viewer.sync()

        # self.data_recording.save()
        # self._plot_ft_comparison(real_wrenches)
        self.run_bayesian_optimization(real_wrenches, joint_traj, viewer=viewer)
        # self.run_pso_optimization(real_wrenches, joint_traj, viewer=viewer, n_iter=self._PSO_ITERATIONS, n_particles=self._PSO_PARTICLES)

    def _plot_ft_comparison(self, real_wrenches: np.ndarray):
        """Plot sim vs real F/T for the last recorded primitive and save as PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.evaluation.quality_metric_utils import butter_lowpass_filter

        sim_data = np.load(self.data_recording.savepath + ".npz")
        sim_time = sim_data["timestamp"]          # (N,)
        sim_fts  = sim_data["eef_fts"].copy()     # (N, 6): Fx Fy Fz Tx Ty Tz
        sim_fts  -= sim_fts[0]

        real_time = real_wrenches[:, 0]           # (M,)
        real_fts  = real_wrenches[:, 1:7].copy()  # (M, 6)
        real_fts  -= real_fts[0]

        cutoff = 33.0
        fs_sim  = 1.0 / self._sim_timestep
        fs_real = 1.0 / float(np.mean(np.diff(real_time)))
        for j in range(6):
            sim_fts[:, j]  = butter_lowpass_filter(sim_fts[:, j],  cutoff=cutoff, fs=fs_sim,  order=4)
            real_fts[:, j] = butter_lowpass_filter(real_fts[:, j], cutoff=cutoff, fs=fs_real, order=4)

        channel_labels = ["Fx [N]", "Fy [N]", "Fz [N]", "Tx [Nm]", "Ty [Nm]", "Tz [Nm]"]
        fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=False)
        axs = axs.flatten()

        for j in range(6):
            axs[j].plot(sim_time,  sim_fts[:, j],  label="sim",  linewidth=1.0)
            axs[j].plot(real_time, real_fts[:, j], label="real", linewidth=1.0, alpha=0.75)
            axs[j].set_title(channel_labels[j])
            axs[j].set_xlabel("t [s]")
            axs[j].grid(True, alpha=0.3)

        axs[0].legend()
        plt.suptitle("Sim vs Real — F/T comparison")
        plt.tight_layout()

        save_path = self.data_recording.savepath + "_ft_comparison.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"F/T comparison plot saved to {save_path}")

    def _pso_prebuild_models(self) -> dict:
        """
        Build one compiled MuJoCo binary per (peg_mesh, hole_mesh) combination.
        Returns {(peg_idx, hole_idx): path}.
        """
        pso_dir = os.path.join(os.path.dirname(self._compiled_model_path), "pso_models")
        os.makedirs(pso_dir, exist_ok=True)

        model_paths = {}
        orig_compiled_path = self._compiled_model_path

        for peg_idx, peg_type in enumerate(self._PSO_PEG_MESH_TYPES):
            for hole_idx, hole_type in enumerate(self._PSO_HOLE_MESH_TYPES):
                key = (peg_idx, hole_idx)
                cache_path = os.path.join(pso_dir, f"pso_model_peg_{peg_type}_hole_{hole_type}.mjb")
                model_paths[key] = cache_path

                if os.path.exists(cache_path):
                    print(f"[PSO] Model peg='{peg_type}' hole='{hole_type}' already cached.")
                    continue

                print(f"[PSO] Building model peg='{peg_type}' hole='{hole_type}' ...")
                modified_task_objs = copy.deepcopy(self._config.get('task_objects', []))
                modified_task_objs[0]['mesh_type'] = peg_type
                modified_task_objs[1]['mesh_type'] = hole_type

                self._compiled_model_path = cache_path
                self._mj_spec = self.load_robot(self._config['robot'])
                self.load_task_objects(modified_task_objs)
                self.compile_model()

        self._compiled_model_path = orig_compiled_path
        return model_paths

    def _pso_evaluate(
        self,
        particle: np.ndarray,
        model_paths: dict,
        joint_traj: np.ndarray,
        real_fts_filtered: np.ndarray,
        real_time: np.ndarray,
        nominal_masses: dict,
        viewer=None,
        viewer_model=None,
        viewer_data=None,
    ) -> float:
        """Simulate one parameter set and return the RMS F/T error vs real data."""
        from src.evaluation.quality_metric_utils import butter_lowpass_filter

        peg_idx  = int(np.clip(round(particle[0]), 0, len(self._PSO_PEG_MESH_TYPES)  - 1))
        hole_idx = int(np.clip(round(particle[1]), 0, len(self._PSO_HOLE_MESH_TYPES) - 1))
        density  = float(particle[2])
        solref   = particle[3:5]
        solimp   = particle[5:10]
        friction = particle[10:13]

        # Swap to the pre-built model for this (peg, hole) mesh combination
        self._mj_model = mujoco.MjModel.from_binary_path(model_paths[(peg_idx, hole_idx)])
        self._mj_model.opt.timestep = self._sim_timestep
        self._mj_data = mujoco.MjData(self._mj_model)
        self.robot.update_mj_pointer(self._mj_model, self._mj_data)

        # Ensure correct contact bitmask (mirrors exec_sim)
        for i in range(self._mj_model.ngeom):
            if self._mj_model.geom_conaffinity[i] != 0:
                self._mj_model.geom_conaffinity[i] = 5

        # Update contact params and scale body masses for density in-place
        mass_scale = density / self._PSO_NOMINAL_DENSITY
        task_body_names = [obj['obj_name'] + '_body' for obj in self._config.get('task_objects', [])]
        for body_name in task_body_names:
            body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            nom = nominal_masses.get(body_name, {}).get((peg_idx, hole_idx))
            if nom is not None:
                self._mj_model.body_mass[body_id] = nom * mass_scale
            for gid in range(self._mj_model.ngeom):
                if self._mj_model.geom_bodyid[gid] == body_id:
                    self._mj_model.geom_solref[gid, :2]  = solref
                    self._mj_model.geom_solimp[gid, :5]  = solimp
                    self._mj_model.geom_friction[gid, :3] = friction

        # Reset: start at the initial joint configuration
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        self._mj_data.qpos[0:6] = joint_traj[0]
        self._mj_data.ctrl[0:6] = joint_traj[0]
        mujoco.mj_forward(self._mj_model, self._mj_data)
        for _ in range(500):
            mujoco.mj_step(self._mj_model, self._mj_data)

        # Replay trajectory and collect FT
        n_steps = len(joint_traj)
        sim_fts = np.zeros((n_steps, 6))
        for i in range(n_steps):
            self._mj_data.ctrl[0:6] = joint_traj[i]
            mujoco.mj_step(self._mj_model, self._mj_data)
            sim_fts[i] = self.robot.get_fts_data(transform_to_base=False)
            if viewer is not None and viewer_data is not None and i % self.iterations_per_frame == 0:
                # Copy joint state back into the viewer's original data so the
                # passive viewer (bound to the launch-time model) stays live.
                nq = min(viewer_data.qpos.shape[0], self._mj_data.qpos.shape[0])
                viewer_data.qpos[:nq] = self._mj_data.qpos[:nq]
                mujoco.mj_kinematics(viewer_model, viewer_data)
                viewer.sync()

        # Offset-correct and filter
        sim_fts -= sim_fts[0]
        # fs_sim = 1.0 / self._sim_timestep
        # for j in range(6):
        #     sim_fts[:, j] = butter_lowpass_filter(sim_fts[:, j], cutoff=33.0, fs=fs_sim, order=4)

        # Interpolate real FT to the sim time grid
        sim_time = np.arange(n_steps) * self._sim_timestep
        real_interp = np.zeros_like(sim_fts)
        for j in range(6):
            real_interp[:, j] = np.interp(
                sim_time, real_time, real_fts_filtered[:, j],
                left=real_fts_filtered[0, j], right=real_fts_filtered[-1, j],
            )

        loss = float(np.sqrt(np.mean((sim_fts - real_interp) ** 2)))
        return loss, sim_fts, sim_time

    def _pso_plot_live(
        self,
        iteration: int,
        n_iter: int,
        real_time: np.ndarray,
        real_fts: np.ndarray,
        sim_time: np.ndarray,
        sim_fts: np.ndarray,
        gbest: np.ndarray,
        convergence: list,
        save_path: str,
    ):
        """Save a live-updating PSO progress figure (overwrites the same file each iteration)."""
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        peg_name  = self._PSO_PEG_MESH_TYPES[int(np.clip(round(gbest[0]),  0, len(self._PSO_PEG_MESH_TYPES)  - 1))]
        hole_name = self._PSO_HOLE_MESH_TYPES[int(np.clip(round(gbest[1]), 0, len(self._PSO_HOLE_MESH_TYPES) - 1))]
        channel_labels = ["Fx [N]", "Fy [N]", "Fz [N]", "Tx [Nm]", "Ty [Nm]", "Tz [Nm]"]

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], hspace=0.45, wspace=0.35)

        # F/T subplots (2×3)
        for j in range(6):
            ax = fig.add_subplot(gs[j // 3, j % 3])
            ax.plot(real_time, real_fts[:, j], color="tab:gray",  linewidth=1.0, label="real", alpha=0.8)
            if sim_fts is not None:
                ax.plot(sim_time,  sim_fts[:, j],  color="tab:blue", linewidth=1.2, label="sim (best)")
            ax.set_title(channel_labels[j], fontsize=9)
            ax.set_xlabel("t [s]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7)

        # Convergence subplot (bottom-left + bottom-middle)
        ax_conv = fig.add_subplot(gs[2, :2])
        ax_conv.plot(np.arange(1, len(convergence) + 1), convergence, marker="o", markersize=3, color="tab:orange")
        ax_conv.set_xlabel("Iteration", fontsize=8)
        ax_conv.set_ylabel("Best RMS loss", fontsize=8)
        ax_conv.set_title("Convergence", fontsize=9)
        ax_conv.tick_params(labelsize=7)
        ax_conv.grid(True, alpha=0.3)

        # Parameter text box (bottom-right)
        ax_txt = fig.add_subplot(gs[2, 2])
        ax_txt.axis("off")
        param_lines = [
            f"Iter {iteration+1}/{n_iter}   loss={convergence[-1]:.5f}",
            f"peg_mesh  : {peg_name}",
            f"hole_mesh : {hole_name}",
            f"density   : {gbest[2]:.1f} kg/m³",
            f"solref    : [{gbest[3]:.4f}, {gbest[4]:.3f}]",
            f"solimp    : [{gbest[5]:.3f}, {gbest[6]:.3f}, {gbest[7]:.5f},",
            f"             {gbest[8]:.3f}, {gbest[9]:.2f}]",
            f"friction  : [{gbest[10]:.3f}, {gbest[11]:.4f}, {gbest[12]:.5f}]",
        ]
        ax_txt.text(
            0.02, 0.95, "\n".join(param_lines),
            transform=ax_txt.transAxes,
            fontsize=7.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        fig.suptitle(f"PSO live — iteration {iteration+1}/{n_iter}", fontsize=11)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def run_pso_optimization(
        self,
        real_wrenches: np.ndarray,
        joint_traj: np.ndarray,
        n_particles: int = 20,
        n_iter: int = None,
        viewer=None,
    ) -> dict:
        """
        Particle Swarm Optimisation over 13 simulation parameters via pyswarms.

        Parameters optimised  (13D)
        --------------------
        peg_mesh_idx   – index into _PSO_PEG_MESH_TYPES  (0–2)
        hole_mesh_idx  – index into _PSO_HOLE_MESH_TYPES (0–1)
        density        – material density in kg/m³ (scales body masses)
        solref[2]      – [timeconst, dampratio]
        solimp[5]      – [d0, dwidth, width, midpoint, power]
        friction[3]    – [sliding, torsional, rolling]

        Returns the best parameter dict and saves it + a convergence plot.
        """
        from pyswarms.single import GlobalBestPSO
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.evaluation.quality_metric_utils import butter_lowpass_filter

        if n_iter is None:
            n_iter = self._PSO_ITERATIONS

        # --- prepare filtered real FT ---
        real_time = real_wrenches[:, 0]
        real_fts = real_wrenches[:, 1:7].copy()
        real_fts -= real_fts[0]
        fs_real = 1.0 / float(np.mean(np.diff(real_time)))
        for j in range(6):
            real_fts[:, j] = butter_lowpass_filter(real_fts[:, j], cutoff=33.0, fs=fs_real, order=4)

        # --- pre-build models ---
        model_paths = self._pso_prebuild_models()

        # --- collect nominal body masses (one per (peg_idx, hole_idx) × body) ---
        task_body_names = [obj['obj_name'] + '_body' for obj in self._config.get('task_objects', [])]
        nominal_masses: dict = {}
        for key, path in model_paths.items():
            mdl = mujoco.MjModel.from_binary_path(path)
            for body_name in task_body_names:
                bid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid >= 0:
                    nominal_masses.setdefault(body_name, {})[key] = float(mdl.body_mass[bid])

        orig_compiled_path = self._compiled_model_path
        viewer_model = self._mj_model if viewer is not None else None
        viewer_data  = self._mj_data  if viewer is not None else None

        result_dir = self.data_recording.RESULT_DIR
        os.makedirs(result_dir, exist_ok=True)
        live_plot_path = os.path.join(result_dir, "pso_live.png")

        # Mutable state shared with the cost-function closure
        state = {
            'iter':        0,
            'gbest_score': np.inf,
            'gbest_pos':   None,
            'gbest_fts':   None,
            'gbest_time':  None,
            'convergence': [],
        }

        def cost_fn(X):
            """Evaluate all particles in the swarm; called once per iteration."""
            costs = np.zeros(len(X))
            for p, particle in enumerate(X):
                score, sim_fts, sim_time = self._pso_evaluate(
                    particle, model_paths, joint_traj,
                    real_fts, real_time, nominal_masses,
                    viewer=viewer, viewer_model=viewer_model, viewer_data=viewer_data,
                )
                costs[p] = score
                if score < state['gbest_score']:
                    state['gbest_score'] = score
                    state['gbest_pos']   = particle.copy()
                    state['gbest_fts']   = sim_fts.copy()
                    state['gbest_time']  = sim_time.copy()

            state['convergence'].append(state['gbest_score'])
            best_peg  = self._PSO_PEG_MESH_TYPES[int(np.clip(round(state['gbest_pos'][0]),  0, len(self._PSO_PEG_MESH_TYPES)  - 1))]
            best_hole = self._PSO_HOLE_MESH_TYPES[int(np.clip(round(state['gbest_pos'][1]), 0, len(self._PSO_HOLE_MESH_TYPES) - 1))]
            print(f"[PSO] iter {state['iter']+1:3d}/{n_iter}  loss={state['gbest_score']:.5f}  peg={best_peg}  hole={best_hole}  density={state['gbest_pos'][2]:.1f}")
            self._pso_plot_live(
                state['iter'], n_iter,
                real_time, real_fts,
                state['gbest_time'], state['gbest_fts'],
                state['gbest_pos'], state['convergence'], live_plot_path,
            )
            state['iter'] += 1
            return costs

        print(f"[PSO] {n_particles} particles × {n_iter} iterations, dim={len(self._PSO_BOUNDS_LO)}")
        optimizer = GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(self._PSO_BOUNDS_LO),
            options={'c1': 1.5, 'c2': 1.5, 'w': 0.7},
            bounds=(self._PSO_BOUNDS_LO, self._PSO_BOUNDS_HI),
        )
        optimizer.optimize(cost_fn, iters=n_iter, verbose=False)
        gbest = state['gbest_pos']

        # --- restore original compiled model ---
        self._compiled_model_path = orig_compiled_path
        self._mj_model = mujoco.MjModel.from_binary_path(self._compiled_model_path)
        self._mj_model.opt.timestep = self._sim_timestep
        self._mj_data = mujoco.MjData(self._mj_model)
        self.robot.update_mj_pointer(self._mj_model, self._mj_data)
        self._mj_renderer = mujoco.Renderer(self._mj_model, height=720, width=1280)

        # --- save results ---
        best_peg  = self._PSO_PEG_MESH_TYPES[int(np.clip(round(gbest[0]),  0, len(self._PSO_PEG_MESH_TYPES)  - 1))]
        best_hole = self._PSO_HOLE_MESH_TYPES[int(np.clip(round(gbest[1]), 0, len(self._PSO_HOLE_MESH_TYPES) - 1))]
        best_params = {
            'peg_mesh_type':  best_peg,
            'hole_mesh_type': best_hole,
            'density':        float(gbest[2]),
            'solref':         gbest[3:5].tolist(),
            'solimp':         gbest[5:10].tolist(),
            'friction':       gbest[10:13].tolist(),
            'best_loss':      float(state['gbest_score']),
        }

        params_path = os.path.join(result_dir, "pso_best_params.yaml")
        with open(params_path, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
        print(f"[PSO] Best params saved to {params_path}")
        print(f"[PSO] Best: peg={best_peg}, hole={best_hole}, loss={state['gbest_score']:.5f}")
        print(f"[PSO]   density={gbest[2]:.1f} kg/m³")
        print(f"[PSO]   solref={gbest[3:5].tolist()}")
        print(f"[PSO]   solimp={gbest[5:10].tolist()}")
        print(f"[PSO]   friction={gbest[10:13].tolist()}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, len(state['convergence']) + 1), state['convergence'], marker='o', markersize=3)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best RMS F/T error")
        ax.set_title("PSO convergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        conv_path = os.path.join(result_dir, "pso_convergence.png")
        plt.savefig(conv_path, dpi=150)
        plt.close()
        print(f"[PSO] Convergence plot saved to {conv_path}")

        return best_params

    def run_bayesian_optimization(
        self,
        real_wrenches: np.ndarray,
        joint_traj: np.ndarray,
        n_calls: int = 200,
        n_initial_points: int = 10,
        viewer=None,
    ) -> dict:
        """
        Bayesian optimisation (Gaussian-process surrogate) over the same 13
        simulation parameters as run_pso_optimization, using scikit-optimize's
        gp_minimize.

        n_calls          – total number of objective evaluations
        n_initial_points – random evaluations before the GP surrogate is used
        """
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.evaluation.quality_metric_utils import butter_lowpass_filter

        # --- prepare filtered real FT ---
        real_time = real_wrenches[:, 0]
        real_fts = real_wrenches[:, 1:7].copy()
        real_fts -= real_fts[0]
        fs_real = 1.0 / float(np.mean(np.diff(real_time)))
        for j in range(6):
            real_fts[:, j] = butter_lowpass_filter(real_fts[:, j], cutoff=33.0, fs=fs_real, order=4)

        # --- pre-build models ---
        model_paths = self._pso_prebuild_models()

        # --- collect nominal body masses ---
        task_body_names = [obj['obj_name'] + '_body' for obj in self._config.get('task_objects', [])]
        nominal_masses: dict = {}
        for key, path in model_paths.items():
            mdl = mujoco.MjModel.from_binary_path(path)
            for body_name in task_body_names:
                bid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid >= 0:
                    nominal_masses.setdefault(body_name, {})[key] = float(mdl.body_mass[bid])

        orig_compiled_path = self._compiled_model_path
        viewer_model = self._mj_model if viewer is not None else None
        viewer_data  = self._mj_data  if viewer is not None else None

        result_dir = self.data_recording.RESULT_DIR
        os.makedirs(result_dir, exist_ok=True)
        live_plot_path = os.path.join(result_dir, "pso_live.png")

        # Search space — Integer for mesh-type indices, Real for everything else
        lo, hi = self._PSO_BOUNDS_LO, self._PSO_BOUNDS_HI
        dimensions = [
            Integer(0, len(self._PSO_PEG_MESH_TYPES)  - 1, name='peg_mesh_idx'),
            Integer(0, len(self._PSO_HOLE_MESH_TYPES) - 1, name='hole_mesh_idx'),
            Real(lo[2],  hi[2],  name='density'),
            Real(lo[3],  hi[3],  name='solref_0'),
            Real(lo[4],  hi[4],  name='solref_1'),
            Real(lo[5],  hi[5],  name='solimp_0'),
            Real(lo[6],  hi[6],  name='solimp_1'),
            Real(lo[7],  hi[7],  name='solimp_2'),
            Real(lo[8],  hi[8],  name='solimp_3'),
            Real(lo[9],  hi[9],  name='solimp_4'),
            Real(lo[10], hi[10], name='friction_0'),
            Real(lo[11], hi[11], name='friction_1'),
            Real(lo[12], hi[12], name='friction_2'),
        ]

        state = {
            'call':        0,
            'gbest_score': np.inf,
            'gbest_pos':   None,
            'gbest_fts':   None,
            'gbest_time':  None,
            'convergence': [],
        }

        def objective(x):
            particle = np.array(x, dtype=float)
            score, sim_fts, sim_time = self._pso_evaluate(
                particle, model_paths, joint_traj,
                real_fts, real_time, nominal_masses,
                viewer=viewer, viewer_model=viewer_model, viewer_data=viewer_data,
            )
            if score < state['gbest_score']:
                state['gbest_score'] = score
                state['gbest_pos']   = particle.copy()
                state['gbest_fts']   = sim_fts.copy()
                state['gbest_time']  = sim_time.copy()
            state['call'] += 1
            return score

        def on_step(res):
            state['convergence'].append(state['gbest_score'])
            best_peg  = self._PSO_PEG_MESH_TYPES[int(np.clip(state['gbest_pos'][0],  0, len(self._PSO_PEG_MESH_TYPES)  - 1))]
            best_hole = self._PSO_HOLE_MESH_TYPES[int(np.clip(state['gbest_pos'][1], 0, len(self._PSO_HOLE_MESH_TYPES) - 1))]
            print(f"[Bayes] call {state['call']:3d}/{n_calls}  loss={state['gbest_score']:.5f}  peg={best_peg}  hole={best_hole}  density={state['gbest_pos'][2]:.1f}")
            self._pso_plot_live(
                state['call'] - 1, n_calls,
                real_time, real_fts,
                state['gbest_time'], state['gbest_fts'],
                state['gbest_pos'], state['convergence'], live_plot_path,
            )

        print(f"[Bayes] {n_calls} calls ({n_initial_points} random), dim={len(dimensions)}")
        gp_minimize(
            objective,
            dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func='gp_hedge',
            random_state=42,
            verbose=False,
            callback=on_step,
        )
        gbest = state['gbest_pos']

        # --- restore original compiled model ---
        self._compiled_model_path = orig_compiled_path
        self._mj_model = mujoco.MjModel.from_binary_path(self._compiled_model_path)
        self._mj_model.opt.timestep = self._sim_timestep
        self._mj_data = mujoco.MjData(self._mj_model)
        self.robot.update_mj_pointer(self._mj_model, self._mj_data)
        self._mj_renderer = mujoco.Renderer(self._mj_model, height=720, width=1280)

        # --- save results ---
        best_peg  = self._PSO_PEG_MESH_TYPES[int(np.clip(gbest[0],  0, len(self._PSO_PEG_MESH_TYPES)  - 1))]
        best_hole = self._PSO_HOLE_MESH_TYPES[int(np.clip(gbest[1], 0, len(self._PSO_HOLE_MESH_TYPES) - 1))]
        best_params = {
            'peg_mesh_type':  best_peg,
            'hole_mesh_type': best_hole,
            'density':        float(gbest[2]),
            'solref':         gbest[3:5].tolist(),
            'solimp':         gbest[5:10].tolist(),
            'friction':       gbest[10:13].tolist(),
            'best_loss':      float(state['gbest_score']),
        }

        params_path = os.path.join(result_dir, "pso_best_params.yaml")
        with open(params_path, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
        print(f"[Bayes] Best params saved to {params_path}")
        print(f"[Bayes] Best: peg={best_peg}, hole={best_hole}, loss={state['gbest_score']:.5f}")
        print(f"[Bayes]   density={gbest[2]:.1f} kg/m³")
        print(f"[Bayes]   solref={gbest[3:5].tolist()}")
        print(f"[Bayes]   solimp={gbest[5:10].tolist()}")
        print(f"[Bayes]   friction={gbest[10:13].tolist()}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, len(state['convergence']) + 1), state['convergence'], marker='o', markersize=3)
        ax.set_xlabel("Call")
        ax.set_ylabel("Best RMS F/T error")
        ax.set_title("Bayesian optimisation convergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        conv_path = os.path.join(result_dir, "pso_convergence.png")
        plt.savefig(conv_path, dpi=150)
        plt.close()
        print(f"[Bayes] Convergence plot saved to {conv_path}")

        return best_params

    def exec_sim(self):
        """
        Main function to execute the LABIT benchmark task.
        """
        signal.signal(signal.SIGINT, self.signal_handler)

        with mujoco.viewer.launch_passive(self._mj_model, self._mj_data, show_left_ui=False, show_right_ui=False) as viewer:
            self.update_view_opt(viewer)
            update_view_camera_parameter(viewer, view_type="labit_benchmark")
            viewer.sync()

            # set every object that is not "no-collision" to have conaffinity bits set to 5 to include bit 1 and 4 (101)
            for i in range(self._mj_model.ngeom):
                if self._mj_model.geom_conaffinity[i] != 0:
                    self._mj_model.geom_conaffinity[i] = 5

            # initial home pose
            self._mj_data.ctrl[0:6] = [-1.63, -1.63, 1.89, -1.88, -1.57, 2.92]
            for _ in range(2000):
                self.step_mj_simulation()
                if viewer is not None:
                    viewer.sync()

            print("simulation timestep: {}".format(self._mj_model.opt.timestep))
            self.simulation_task(viewer=viewer)

            viewer.close()

    def signal_handler(self, sig, frame):
        print("\n[EXIT] Benchmark interrupted. Saving current primitive data.")
        try:
            self.data_recording.save()
        except Exception as e:
            print(f"Error saving data: {e}")
        os._exit(0)

    def exec_sim_headless(self):
        signal.signal(signal.SIGINT, self.signal_handler)

        for i in range(self._mj_model.ngeom):
            if self._mj_model.geom_conaffinity[i] != 0:
                self._mj_model.geom_conaffinity[i] = 5

        self._mj_data.ctrl[0:6] = [-1.63, -1.63, 1.89, -1.88, -1.57, 2.92]
        for _ in range(2000):
            self.step_mj_simulation()

        print("simulation timestep: {}".format(self._mj_model.opt.timestep))
        self.simulation_task()

    def exclude_pair_runtime(self, body1_name, body2_name):
        b1_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        b2_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body2_name)

        # 1. Isolate the Box (Body 1)
        # We set it to belong to Layer 2, and look for Layer 1 (others)
        for i in range(self._mj_model.ngeom):
            if self._mj_model.geom_bodyid[i] == b1_id:
                self._mj_model.geom_contype[i] = 2      # I am Layer 2
                self._mj_model.geom_conaffinity[i] = 1  # I hit Layer 1 (others)

        # 2. Isolate the Lid (Body 2)
        # We set it to belong to Layer 4, and look for Layer 1 (others)
        for i in range(self._mj_model.ngeom):
            if self._mj_model.geom_bodyid[i] == b2_id:
                self._mj_model.geom_contype[i] = 4      # I am Layer 4
                self._mj_model.geom_conaffinity[i] = 1  # I hit Layer 1 (others)


if __name__ == "__main__":

    for _ in range(NUM_RUNS):
        mj = PositionBasedInsertion(
            task_env_config_path=ENV_CONFIG_PATH,
            server_modus=True,
            sim_timestep=SIM_TIMESTEP,
            )
        # mj.exec_sim()
        mj.exec_sim_headless()
    os._exit(0)
