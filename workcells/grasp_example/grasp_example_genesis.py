"""Minimal Genesis example: UR5e robot grasps an object and lifts it."""
import os, signal, time, tempfile, re
import numpy as np
import yaml
import genesis as gs
from scipy.spatial.transform import Rotation as R


CONFIG_PATH = "/workspace/workcells/grasp_example/grasp_example.yaml"
ROBOT_MJCF  = "/workspace/src/assets/robots/ur5e/ur5e_genesis.xml"

SIM_DT   = 0.001
TOOL_LEN = 0.21   # distance from tool0 origin to fingertip contact along tool0 z-axis [m]

ARM_JOINTS     = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
GRIPPER_JOINTS = ["hande_left_finger_joint", "hande_right_finger_joint"]

# Home configuration (arm up, ready to reach forward)
HOME_Q = [0.0, -1.2, 1.5, -1.9, -1.57, 0.0]


# ── helpers ────────────────────────────────────────────────────────────────────

def _xyzw(q): return np.array([q[1], q[2], q[3], q[0]])
def _wxyz(q): return np.array([q[3], q[0], q[1], q[2]])

def _grasp_quat(obj_quat_wxyz):
    """EEF orientation derived from object frame with z-axis forced downward.

    Takes the object's rotation matrix and flips it (180° around its x-axis)
    when its z-axis points up in the world frame, so the gripper always
    descends along the object's -z direction.
    """
    mat = R.from_quat(_xyzw(obj_quat_wxyz)).as_matrix()
    if mat[2, 2] > 0:          # object z-axis points up → flip to point down
        mat = mat @ np.diag([1., -1., -1.])
    return _wxyz(R.from_matrix(mat).as_quat())


def _clean_robot_mjcf(src_path):
    """Strip <equality> block from MJCF so Genesis doesn't choke on weld constraints."""
    with open(src_path) as f:
        xml = f.read()
    xml = re.sub(r'\s*<equality>.*?</equality>', '', xml, flags=re.DOTALL)
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w",
                                     dir=os.path.dirname(src_path))
    tmp.write(xml)
    tmp.close()
    return tmp.name


# ── Simulation ─────────────────────────────────────────────────────────────────

class GraspExample:

    def __init__(self, headless=True):

        if not headless:
            if not os.environ.get("DISPLAY"):
                print("[LabItGenesis] No DISPLAY found — falling back to headless mode.")
                headless = True
            else:
                # Genesis viewer uses pyglet/GLX (X11); EGL set globally conflicts with it
                os.environ["PYOPENGL_PLATFORM"] = "glx"

        gs.init(backend=gs.cpu, logging_level="warning")

        with open(CONFIG_PATH) as f:
            self._cfg = yaml.safe_load(f)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=SIM_DT,
                substeps=5,
                gravity=(0, 0, -9.81),
            ),
            show_viewer=not headless,
        )

        self._build_scene()
        self.scene.build()
        self._init_robot()

    # ── scene construction ─────────────────────────────────────────────────────

    def _build_scene(self):
        self.scene.add_entity(gs.morphs.Plane())
        self._load_robot()
        self._load_objects()

    def _load_robot(self):
        rp   = self._cfg["robot"]["base_pose"]
        pos  = np.array(rp["position"])
        quat = np.array(rp["quaternion"])
        clean = _clean_robot_mjcf(ROBOT_MJCF)
        try:
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=clean,
                    pos=pos,
                    quat=quat,
                    requires_jac_and_IK=True,
                    recompute_inertia=True,
                )
            )
        finally:
            os.remove(clean)

    def _load_objects(self):
        self._entities = {}
        for obj in self._cfg.get("task_objects", []):
            name      = obj["obj_name"]
            mesh_path = obj.get("mesh_path")
            pos       = np.array(obj["attach_pose"]["position"])
            quat      = np.array(obj["attach_pose"]["quaternion"])
            scale     = obj.get("scale", [1, 1, 1])
            is_free   = obj.get("joint") == "free"
            color     = obj.get("mesh_color")

            surface = gs.surfaces.Default(color=tuple(color)) if color else None

            ent = self.scene.add_entity(
                gs.morphs.Mesh(
                    file=mesh_path,
                    pos=pos,
                    quat=quat,
                    scale=float(scale[0]),
                    fixed=not is_free,
                ),
                surface=surface,
            )
            self._entities[name] = ent

    # ── robot initialisation ───────────────────────────────────────────────────

    def _init_robot(self):
        self._arm_dofs  = [self.robot.get_joint(j).dof_idx_local for j in ARM_JOINTS]
        self._grip_dofs = [self.robot.get_joint(j).dof_idx_local for j in GRIPPER_JOINTS]
        self._eef_link  = self.robot.get_link("tool0")

        self.robot.set_dofs_kp(np.full(6, 2000.), self._arm_dofs)
        self.robot.set_dofs_kv(np.full(6, 400.),  self._arm_dofs)
        self.robot.set_dofs_kp(np.array([300., 300.]), self._grip_dofs)
        self.robot.set_dofs_kv(np.array([50.,  50.]),  self._grip_dofs)

    # ── motion primitives ──────────────────────────────────────────────────────

    def move_to_joint_pos(self, q_goal, timeout=4000):
        q_goal = np.asarray(q_goal, dtype=float)
        self.robot.control_dofs_position(q_goal, self._arm_dofs)
        for _ in range(timeout):
            self.scene.step()
            if np.linalg.norm(
                self.robot.get_dofs_position(self._arm_dofs).numpy() - q_goal
            ) < 0.01:
                break

    def _ik(self, pos, quat, seed_q=None):
        """Compute IK without physical motion. Temporarily seeds from seed_q if given."""
        if seed_q is not None:
            saved = self.robot.get_dofs_position(self._arm_dofs).numpy().copy()
            self.robot.set_dofs_position(np.asarray(seed_q), self._arm_dofs)
        q = self.robot.inverse_kinematics(
            link=self._eef_link,
            pos=pos,
            quat=quat,
            dofs_idx_local=self._arm_dofs,
        ).numpy()[self._arm_dofs]
        if seed_q is not None:
            self.robot.set_dofs_position(saved, self._arm_dofs)
        return q

    def move_eef(self, target_pos, target_quat_wxyz, timeout=5000, seed_q=None):
        q = self._ik(target_pos, target_quat_wxyz, seed_q=seed_q)
        self.robot.control_dofs_position(q, self._arm_dofs)
        for _ in range(timeout):
            self.scene.step()
            if np.linalg.norm(
                self.robot.get_dofs_position(self._arm_dofs).numpy() - q
            ) < 0.01:
                break

    def set_gripper(self, opening_m, timeout=2000):
        target = float(np.clip(0.025 - opening_m / 2, 0., 0.025))
        tgt = np.array([target, target])
        self.robot.control_dofs_position(tgt, self._grip_dofs)
        for _ in range(timeout):
            self.scene.step()
            cur = self.robot.get_dofs_position(self._grip_dofs).numpy()
            vel = self.robot.get_dofs_velocity(self._grip_dofs).numpy()
            if abs(cur[0] - target) < 1e-4 or (abs(vel[0]) < 1e-4 and abs(vel[1]) < 1e-4):
                break

    # ── grasp policy ───────────────────────────────────────────────────────────

    def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)

        # Physics warm-up
        for _ in range(500):
            self.scene.step()

        obj = self._entities["cube"]
        obj_pos  = obj.get_pos().numpy()
        obj_quat = obj.get_quat().numpy()   # wxyz from Genesis

        # Grasp orientation: object frame with z-axis forced to point downward.
        # tool0 z-axis is the approach direction (toward fingertips); IK is asked
        # to orient tool0 so this axis points in world -z → gripper descends from above.
        grasp_quat = _grasp_quat(obj_quat)

        # All positions below are for the tool0 ORIGIN.
        # Fingertips are TOOL_LEN below tool0 along its z-axis (world -z for this quat).
        # Add TOOL_LEN to every z-offset so the fingers land at the intended height.
        pre_grasp_pos = obj_pos + np.array([0., 0., TOOL_LEN + 0.15])   # 15 cm clearance above obj
        grasp_pos     = obj_pos + np.array([0., 0., TOOL_LEN + 0.005])  # fingertips 5 mm above obj
        lift_pos      = obj_pos + np.array([0., 0., TOOL_LEN + 0.25])   # lift 25 cm above obj

        print("[1] Moving to home position")
        self.move_to_joint_pos(HOME_Q)

        # Compute grasp IK from HOME config so pre-grasp uses the same solution family.
        q_grasp_ref = self._ik(grasp_pos, grasp_quat)

        print("[2] Opening gripper")
        self.set_gripper(opening_m=0.04)

        print("[3] Moving to pre-grasp (above object)")
        self.move_eef(pre_grasp_pos, grasp_quat, seed_q=q_grasp_ref)

        print("[4] Moving to grasp pose")
        self.move_eef(grasp_pos, grasp_quat)

        print("[5] Closing gripper")
        self.set_gripper(opening_m=0.0)

        print("[6] Lifting object")
        self.move_eef(lift_pos, grasp_quat)

        # Report result
        final_pos = obj.get_pos().numpy()
        lifted = final_pos[2] - obj_pos[2]
        print(f"[DONE] Object lifted by {lifted:.3f} m "
              f"(final z = {final_pos[2]:.3f} m)")

        # Keep simulating so the viewer stays open
        while True:
            self.scene.step()

    def _signal_handler(self, *_):
        print("\n[EXIT] Interrupted.")
        os._exit(0)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--viewer", action="store_true", help="Enable viewer (requires display)")
    args = p.parse_args()
    sim = GraspExample(headless=not args.viewer)
    sim.run()
