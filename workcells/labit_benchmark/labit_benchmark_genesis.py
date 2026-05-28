"""Genesis-based LABIT benchmark simulation."""
import os, signal, time, tempfile, re
import numpy as np
import yaml
import genesis as gs
from scipy.spatial.transform import Rotation as R


CONFIG_PATH  = "/workspace/src/configs/envs/ur5e_labit_benchmark.yaml"
ROBOT_MJCF   = "/workspace/src/assets/robots/ur5e/ur5e_genesis.xml"

# Receiving parts with internal concave geometry (holes, slots, pockets).
# These need detailed CoACD decomposition so insertion geometry is preserved.
COLLISION_FINE = frozenset([
    "housing_bottom",
    "housing_top",
    "housing_middle",
    "rotor_printed_part",
    "screw_m5_16_hexagon_head_1", "screw_m5_16_hexagon_head_2", "screw_m5_16_hexagon_head_3",
    "screw_m5_16_hexagon_head_4", "screw_m5_16_hexagon_head_5",
    "parts_supply",
    "plug_inside_loose_1", "plug_inside_loose_2",
    "plug_outside_loose",
    "cover_plate",
    "pcb",
    "gearwheel_teeth_35_mod_2_1", "gearwheel_teeth_35_mod_2_2",
    "tube",
    "o_ring",
    "tube_clamp",
])

# Near-convex inserting parts (pins, screws, bolts, flat plates, simple solids).
# Skip CoACD entirely — the single convex hull matches the real geometry closely.
COLLISION_NO_CONVEXIFY = frozenset([
    "positioning_pin_d5_20_1", "positioning_pin_d5_20_2", "positioning_pin_d5_20_3",
    "positioning_pin_d5_20_4", "positioning_pin_d5_20_5", "positioning_pin_d5_20_6",
    "bolt_rotor",
    "bolt_middle_housing",
])

# Fine-decomposition CoACD options: low threshold → many small convex pieces so
# holes and slots are faithfully represented for peg-in-hole / insertion tasks.
_COACD_FINE = gs.options.CoacdOptions(threshold=0.04, max_convex_hull=64)

def _surface(rgba):
    """Build a gs.surfaces.Default from an RGBA list, or return None if absent."""
    if rgba is None:
        return None
    r, g, b = rgba[0], rgba[1], rgba[2]
    a = rgba[3] if len(rgba) > 3 else 1.0
    return gs.surfaces.Default(color=(r, g, b, a))


def _make_morph(name: str, mesh_path: str, pos, quat, scale: float, fixed: bool):
    """Build a gs.morphs.Mesh with collision settings tuned for assembly tasks.

    Three tiers:
      COLLISION_FINE       — low CoACD threshold + many hulls; preserves holes/slots
                             in receiving parts (housings, rotor) for insertion tasks.
      COLLISION_NO_CONVEXIFY — skip CoACD; single convex hull is accurate enough
                             for near-convex inserting parts (pins, screws, bolts).
      default              — Genesis defaults for everything else.
    """
    if name in COLLISION_FINE:
        return gs.morphs.Mesh(
            file=mesh_path, pos=pos, quat=quat, scale=scale, fixed=fixed,
            decompose_object_error_threshold=0.01,   # force decomposition
            coacd_options=_COACD_FINE,
        )
    if name in COLLISION_NO_CONVEXIFY:
        return gs.morphs.Mesh(
            file=mesh_path, pos=pos, quat=quat, scale=scale, fixed=fixed,
            convexify=False,
        )
    return gs.morphs.Mesh(
        file=mesh_path, pos=pos, quat=quat, scale=scale, fixed=fixed,
    )


def _clean_robot_mjcf(src_path):
    """Return path to a temp MJCF stripped of scene-specific constraints.

    Removes the entire <equality> block:
    - weld constraints reference bodies outside the robot model
    - finger-coupling equality is redundant (both DOFs are controlled directly)
      and its solref=5e-6 would trigger Genesis's 2*dt constraint-clamping warning

    The temp file is written next to the source so that relative mesh paths
    (meshdir) still resolve correctly. Caller is responsible for deleting it.
    """
    with open(src_path) as f:
        xml = f.read()
    xml = re.sub(r'\s*<equality>.*?</equality>', '', xml, flags=re.DOTALL)
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w",
                                      dir=os.path.dirname(src_path))
    tmp.write(xml)
    tmp.close()
    return tmp.name


def _cleanup_robot_tmpfiles(src_dir):
    """Remove stale tmp*.xml files left in the robot asset directory."""
    for fname in os.listdir(src_dir):
        if fname.startswith("tmp") and fname.endswith(".xml"):
            try:
                os.remove(os.path.join(src_dir, fname))
            except OSError:
                pass


SIM_DT       = 0.0005
TOOL_LEN     = 0.21

ARM_JOINTS     = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
GRIPPER_JOINTS = ["hande_left_finger_joint", "hande_right_finger_joint"]


# ── SE(3) helpers ──────────────────────────────────────────────────────────────

def _xyzw(q): return np.array([q[1], q[2], q[3], q[0]])
def _wxyz(q): return np.array([q[3], q[0], q[1], q[2]])
def _Rmat(q_wxyz): return R.from_quat(_xyzw(q_wxyz)).as_matrix()

def _compose(p_pos, p_quat_wxyz, l_pos, l_quat_wxyz):
    """Compose parent and local SE(3) transforms; all quats wxyz."""
    Rp = _Rmat(p_quat_wxyz)
    Rl = _Rmat(l_quat_wxyz)
    return (np.array(p_pos) + Rp @ np.array(l_pos),
            _wxyz(R.from_matrix(Rp @ Rl).as_quat()))


# ── Simulation class ───────────────────────────────────────────────────────────

class LabItGenesis:

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
                substeps=5,       # effective per-substep dt = 0.0001 s; prevents NaN constraint forces
                gravity=(0, 0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(
                max_collision_pairs=40000,
            ),
            show_viewer=not headless,
        )

        # _wp[name] = (world_pos, world_quat_wxyz) — initial static world poses
        self._wp       = {"world": (np.zeros(3), np.array([1., 0., 0., 0.]))}
        # _entities[name] = RigidEntity — free objects + top-level static meshes
        self._entities = {}
        # _targets[name] = (parent_name, local_pos, local_quat_wxyz)
        # covers: "none" frames AND static-mesh children of free objects
        self._targets  = {}
        # tracks which names are free (joint: free) so children become dynamic targets
        self._free_names = set()

        self._build_scene()
        self.scene.build()
        self._init_robot()

    # ── scene construction ─────────────────────────────────────────────────────

    def _build_scene(self):
        self.scene.add_entity(gs.morphs.Plane())
        self._load_robot()
        self._load_env_objects()
        # The robot MJCF defines a body named "table" at world origin (identity pose).
        # env_objects also contain a visual mesh named "table" at a different pose which
        # overwrites _wp["table"]. Re-seed so task objects that attach to "table" compose
        # with the correct MJCF body pose.
        self._wp["table"] = (np.zeros(3), np.array([1., 0., 0., 0.]))
        self._load_task_objects()

    def _load_robot(self):
        rp = self._cfg["robot"]["base_pose"]
        pos  = np.array(rp["position"])
        quat = np.array(rp["quaternion"])   # wxyz
        robot_dir  = os.path.dirname(ROBOT_MJCF)
        clean_mjcf = _clean_robot_mjcf(ROBOT_MJCF)
        try:
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=clean_mjcf,
                    pos=pos,
                    quat=quat,
                    requires_jac_and_IK=True,
                    recompute_inertia=True,   # fixes dubious CoM warnings for gripper fingers
                )
            )
        finally:
            # Genesis has parsed the file by the time add_entity returns;
            # delete immediately so no stale tmp*.xml files accumulate.
            os.remove(clean_mjcf)
            _cleanup_robot_tmpfiles(robot_dir)
        self._wp["robot_base"] = (pos, quat)

    def _resolve_parent(self, attach_body_str):
        """Strip _body suffix so 'plate_benchmark_body' → 'plate_benchmark'."""
        name = attach_body_str
        if name.endswith("_body"):
            name = name[:-5]
        return name

    def _load_env_objects(self):
        for obj in self._cfg.get("env_objects", []):
            name      = obj["obj_name"]
            scale     = obj.get("scale", [1, 1, 1])
            mesh_path = obj.get("mesh_path")
            parent    = self._resolve_parent(obj.get("attach_body", "world"))
            lpos      = np.array(obj["attach_pose"]["position"])
            lquat     = np.array(obj["attach_pose"]["quaternion"])
            p_pos, p_quat = self._wp.get(parent, (np.zeros(3), np.array([1., 0., 0., 0.])))
            gpos, gquat   = _compose(p_pos, p_quat, lpos, lquat)

            self._wp[name] = (gpos, gquat)
            if mesh_path:
                ent = self.scene.add_entity(
                    _make_morph(name, mesh_path, gpos, gquat, float(scale[0]), fixed=True),
                    surface=_surface(obj.get("mesh_color")),
                )
                self._entities[name] = ent

    def _parent_is_dynamic(self, parent_name):
        """True when parent is a free entity or already a dynamic target."""
        return parent_name in self._free_names or parent_name in self._targets

    def _load_task_objects(self):
        for obj in self._cfg.get("task_objects", []):
            name      = obj["obj_name"]
            parent    = self._resolve_parent(obj.get("attach_body", "world"))
            lpos      = np.array(obj["attach_pose"]["position"])
            lquat     = np.array(obj["attach_pose"]["quaternion"])
            mesh_type = obj.get("mesh_type", "none")
            mesh_path = obj.get("mesh_path")
            scale     = obj.get("scale", [1, 1, 1])
            is_free   = obj.get("joint") == "free"

            # Compose world pose purely from the YAML hierarchy.
            # genesis_pos == attach_pos_world for all objects because MuJoCo's body
            # placement offset (R @ centroid) and the mesh geom offset (-centroid) cancel
            # exactly — so no centroid correction is needed here.
            p_pos, p_quat = self._wp.get(parent, (np.zeros(3), np.array([1., 0., 0., 0.])))
            gpos, gquat = _compose(p_pos, p_quat, lpos, lquat)

            self._wp[name] = (gpos, gquat)

            # "none" objects are pure target frames — track relative to parent.
            if mesh_type == "none":
                self._targets[name] = (parent, lpos, lquat)
                continue

            # Static mesh whose parent chain contains a free object: Genesis has no
            # "child body" concept, so track as a coordinate frame only (no entity).
            if not is_free and self._parent_is_dynamic(parent):
                self._targets[name] = (parent, lpos, lquat)
                continue

            surf = _surface(obj.get("mesh_color"))

            # Free object → Genesis entity without fixed constraint.
            if is_free:
                self._free_names.add(name)
                ent = self.scene.add_entity(
                    _make_morph(name, mesh_path, gpos, gquat, float(scale[0]), fixed=False),
                    surface=surf,
                )
                self._entities[name] = ent
                continue

            # Static mesh with a fully static parent chain → fixed entity.
            ent = self.scene.add_entity(
                _make_morph(name, mesh_path, gpos, gquat, float(scale[0]), fixed=True),
                surface=surf,
            )
            self._entities[name] = ent

    # ── robot initialisation ───────────────────────────────────────────────────

    def _init_robot(self):
        self._arm_dofs  = [self.robot.get_joint(j).dof_idx_local for j in ARM_JOINTS]
        self._grip_dofs = [self.robot.get_joint(j).dof_idx_local for j in GRIPPER_JOINTS]
        self._eef_link  = self.robot.get_link("tool0")

        self.robot.set_dofs_kp(np.full(6, 2000.),  self._arm_dofs)
        self.robot.set_dofs_kv(np.full(6, 400.),   self._arm_dofs)
        self.robot.set_dofs_kp(np.array([300., 300.]), self._grip_dofs)
        self.robot.set_dofs_kv(np.array([50.,  50.]),  self._grip_dofs)

    # ── entity pose queries ────────────────────────────────────────────────────

    def get_entity_pose(self, name):
        """Return (pos_np, quat_wxyz_np) for any named entity or target frame."""
        if name in self._entities:
            ent = self._entities[name]
            return ent.get_pos().numpy(), ent.get_quat().numpy()
        if name in self._targets:
            parent, lpos, lquat = self._targets[name]
            p_pos, p_quat = self.get_entity_pose(parent)
            return _compose(p_pos, p_quat, lpos, lquat)
        if name in self._wp:
            return self._wp[name]
        raise KeyError(f"Unknown entity/target: {name}")

    # ── world EEF target computation ───────────────────────────────────────────

    def _world_eef_target(self, ref_name, pos_offset, euler_offset, ensure_neg_z):
        body_pos, body_quat = self.get_entity_pose(ref_name)
        body_R = _Rmat(body_quat)
        if ensure_neg_z and body_R[2, 2] > 0:
            body_R = body_R @ np.diag([1., -1., -1.])
        rot_off   = R.from_euler("xyz", euler_offset, degrees=True).as_matrix()
        world_pos = body_pos + body_R @ (np.array(pos_offset) + np.array([0., 0., -TOOL_LEN]))
        world_R   = body_R @ rot_off
        return world_pos, _wxyz(R.from_matrix(world_R).as_quat())

    # ── motion primitives ──────────────────────────────────────────────────────

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

    def move_eef(self, world_pos, world_quat, timeout=5000, seed_q=None):
        q = self._ik(world_pos, world_quat, seed_q=seed_q)
        self.robot.control_dofs_position(q, self._arm_dofs)
        for _ in range(timeout):
            self.scene.step()
            if np.linalg.norm(self.robot.get_dofs_position(self._arm_dofs).numpy() - q) < 0.005:
                break

    def move_to_joint_pos(self, q_goal, timeout=4000):
        q_goal = np.asarray(q_goal, dtype=float)
        self.robot.control_dofs_position(q_goal, self._arm_dofs)
        for _ in range(timeout):
            self.scene.step()
            if np.linalg.norm(self.robot.get_dofs_position(self._arm_dofs).numpy() - q_goal) < 0.005:
                break

    def set_gripper(self, opening_m, timeout=2000):
        target = float(np.clip(0.025 - opening_m / 2, 0., 0.025))
        tgt    = np.array([target, target])
        self.robot.control_dofs_position(tgt, self._grip_dofs)
        for _ in range(timeout):
            self.scene.step()
            cur = self.robot.get_dofs_position(self._grip_dofs).numpy()
            vel = self.robot.get_dofs_velocity(self._grip_dofs).numpy()
            if abs(cur[0] - target) < 1e-4 or (abs(vel[0]) < 1e-4 and abs(vel[1]) < 1e-4):
                break

    def _move_to(self, ref, pos_off, euler_off=None, ensure_neg_z=True, timeout=5000, seed_q=None):
        if euler_off is None:
            euler_off = [0., 0., 0.]
        wpos, wquat = self._world_eef_target(ref, pos_off, euler_off, ensure_neg_z)
        self.move_eef(wpos, wquat, timeout, seed_q=seed_q)

    # ── assembly skills ────────────────────────────────────────────────────────

    def insert(self, body, target, pd, ensure_neg_z=True,
               gripper_open=0.017, gripper_close=0.0):
        print(f"[INSERT] {body} → {target}")
        ez = ensure_neg_z
        self.set_gripper(gripper_open)
        grasp_wpos, grasp_wquat = self._world_eef_target(
            body, pd["grasp"].get("position", [0,0,0]), pd["grasp"].get("orientation", [0,0,0]), ez)
        q_grasp_ref = self._ik(grasp_wpos, grasp_wquat)
        self._move_to(body,   pd["pre_grasp"].get("position",  [0,0,0]), pd["pre_grasp"].get("orientation",  [0,0,0]), ez, seed_q=q_grasp_ref)
        self._move_to(body,   pd["grasp"].get("position",      [0,0,0]), pd["grasp"].get("orientation",      [0,0,0]), ez)
        self.set_gripper(gripper_close)
        self._move_to(body,   pd["after_grasp"].get("position",[0,0,0]), pd["after_grasp"].get("orientation",[0,0,0]), ez)
        self._move_to(target, pd["pre_asm"].get("position",    [0,0,0]), pd["pre_asm"].get("orientation",    [0,0,0]), ez)
        self._move_to(target, pd["asm"].get("position",        [0,0,0]), pd["asm"].get("orientation",        [0,0,0]), ez)
        self.set_gripper(gripper_open)
        self._move_to(target, pd["after_asm"].get("position",  [0,0,0]), pd["after_asm"].get("orientation",  [0,0,0]), ez)

    def screw(self, body, target, pd, ensure_neg_z=True,
              gripper_open=0.017, gripper_close=0.0):
        print(f"[SCREW] {body} → {target}")
        ez = ensure_neg_z
        self.set_gripper(gripper_open)
        grasp_wpos, grasp_wquat = self._world_eef_target(
            body, pd["grasp"].get("position", [0,0,0]), [0,0,0], ez)
        q_grasp_ref = self._ik(grasp_wpos, grasp_wquat)
        self._move_to(body,   pd["pre_grasp"].get("position",  [0,0,0]), ensure_neg_z=ez, seed_q=q_grasp_ref)
        self._move_to(body,   pd["grasp"].get("position",      [0,0,0]), ensure_neg_z=ez)
        self.set_gripper(gripper_close)
        self._move_to(body,   pd["after_grasp"].get("position",[0,0,0]), ensure_neg_z=ez)
        self._move_to(target, pd["pre_asm"].get("position",    [0,0,0]), ensure_neg_z=ez)
        self._move_to(target, pd["asm"].get("position",        [0,0,0]), ensure_neg_z=ez)
        # regrasp + screw rotation
        self.set_gripper(0.01)
        self._move_to(body, pd["grasp"].get("position", [0,0,0]), ensure_neg_z=ez)
        step_angle = np.pi
        for _ in range(int(2 * np.pi / step_angle)):
            cur_pos  = self._eef_link.get_pos().numpy()
            cur_quat = self._eef_link.get_quat().numpy()
            new_R    = _Rmat(cur_quat) @ R.from_euler("z", step_angle).as_matrix()
            self.set_gripper(gripper_close)
            self.move_eef(cur_pos, _wxyz(R.from_matrix(new_R).as_quat()))
            self.set_gripper(0.01)
            self.move_eef(cur_pos, cur_quat)
        self._move_to(target, pd["after_asm"].get("position", [0,0,0]), ensure_neg_z=ez)

    def clamp(self, body, target, pd, ensure_neg_z=False,
              gripper_open=0.02, gripper_close=0.01):
        print(f"[CLAMP] {body} → {target}")
        # intermediate safe position
        self.move_eef(np.array([0.35, -0.2, 0.3]), np.array([1., 0., 0., 0.]))
        ez = ensure_neg_z
        self.set_gripper(gripper_open)
        grasp_wpos, grasp_wquat = self._world_eef_target(
            body, pd["grasp"].get("position", [0,0,0]), pd["grasp"].get("orientation", [0,0,0]), ez)
        q_grasp_ref = self._ik(grasp_wpos, grasp_wquat)
        self._move_to(body,   pd["pre_grasp"].get("position",  [0,0,0]), pd["pre_grasp"].get("orientation",  [0,0,0]), ez, seed_q=q_grasp_ref)
        self._move_to(body,   pd["grasp"].get("position",      [0,0,0]), pd["grasp"].get("orientation",      [0,0,0]), ez)
        self.set_gripper(gripper_close)
        self._move_to(body,   pd["after_grasp"].get("position",[0,0,0]), pd["after_grasp"].get("orientation",[0,0,0]), ez)
        # extra lateral clearance move
        self._move_to(body, [0., 0.35, 0.], pd["after_grasp"].get("orientation", [0,0,0]), ez)
        # from here on always ensure_neg_z=True (same as original)
        self._move_to(target, pd["pre_asm"].get("position",   [0,0,0]), ensure_neg_z=True)
        self._move_to(target, pd["asm"].get("position",       [0,0,0]), ensure_neg_z=True)
        self.set_gripper(gripper_open)
        self._move_to(target, pd["after_asm"].get("position", [0,0,0]), ensure_neg_z=True)

    # ── assembly policy ────────────────────────────────────────────────────────

    def labit_policy(self):
        t0 = time.time()

        self.insert("pcb", "housing_middle_pcb_target", gripper_close=0.006,
            pd={"pre_grasp":  {"position": [0, 0, -0.03]},
                "grasp":       {"position": [0, 0, -0.005]},
                "after_grasp": {"position": [0, 0, -0.25]},
                "pre_asm":     {"position": [0, 0, -0.1]},
                "asm":         {"position": [0, 0, -0.015]},
                "after_asm":   {"position": [0, 0.08, -0.18]}})

        self.insert("plug_inside_loose_1", "plug_inside_fixed_1", gripper_close=0.005,
            pd={"pre_grasp":  {"position": [0, 0, -0.03]},
                "grasp":       {"position": [0, 0, -0.005]},
                "after_grasp": {"position": [0, 0, -0.25]},
                "pre_asm":     {"position": [0, 0, -0.1]},
                "asm":         {"position": [0, 0, -0.009]},
                "after_asm":   {"position": [0, 0, -0.1]}})

        self.insert("plug_inside_loose_2", "plug_inside_fixed_2", gripper_close=0.005,
            pd={"pre_grasp":  {"position": [0, 0, -0.03]},
                "grasp":       {"position": [0, 0, -0.005]},
                "after_grasp": {"position": [0, 0, -0.25]},
                "pre_asm":     {"position": [0, 0, -0.1]},
                "asm":         {"position": [0, 0, -0.009]},
                "after_asm":   {"position": [0.1, 0, -0.1]}})

        self.insert("plug_outside_loose", "plug_outside_fixed", gripper_close=0.009,
            pd={"pre_grasp":  {"position": [0, 0, -0.03]},
                "grasp":       {"position": [0, 0, -0.001]},
                "after_grasp": {"position": [0, 0, -0.25]},
                "pre_asm":     {"position": [0, 0, -0.1]},
                "asm":         {"position": [0, 0, -0.001]},
                "after_asm":   {"position": [0, 0, -0.1]}})

        self.insert("positioning_pin_d5_20_2", "housing_bottom_pin_hole_2", gripper_close=0.0035,
            pd={"pre_grasp":  {"position": [0, 0, -0.02]},
                "grasp":       {"position": [0, 0, -0.004]},
                "after_grasp": {"position": [0, 0, -0.2]},
                "pre_asm":     {"position": [0, 0, -0.05]},
                "asm":         {"position": [0, 0, -0.004]},
                "after_asm":   {"position": [0, 0, -0.1]}})

        self.insert("positioning_pin_d5_20_1", "housing_bottom_pin_hole_1", gripper_close=0.0035,
            pd={"pre_grasp":  {"position": [0, 0, -0.02]},
                "grasp":       {"position": [0, 0, -0.004]},
                "after_grasp": {"position": [0, 0, -0.2]},
                "pre_asm":     {"position": [0, 0, -0.05]},
                "asm":         {"position": [0, 0, -0.004]},
                "after_asm":   {"position": [0, 0, -0.1]}})

        self.insert("bolt_rotor", "housing_bottom", gripper_close=0.005,
            pd={"pre_grasp":  {"position": [0, 0, -0.04]},
                "grasp":       {"position": [0, 0, -0.0058]},
                "after_grasp": {"position": [0, 0, -0.2]},
                "pre_asm":     {"position": [0, 0, -0.15]},
                "asm":         {"position": [0, 0, -0.06]},
                "after_asm":   {"position": [0, 0, -0.15]}})

        self.insert("housing_middle_grasp_target", "housing_middle_release_target",
                    ensure_neg_z=False,
            pd={"pre_grasp":  {"position": [0,    0.003, -0.06]},
                "grasp":       {"position": [0,    0.003,  0.024]},
                "after_grasp": {"position": [0.15, 0.003,  0.024]},
                "pre_asm":     {"position": [-0.15, 0.003, 0.024]},
                "asm":         {"position": [-0.001, 0.003, 0.024]},
                "after_asm":   {"position": [-0.001, 0.003, -0.03]}})

        # clear path to avoid collision with housing stack
        self._move_to("housing_middle_release_target", [-0.25,  0.003, -0.03], ensure_neg_z=False)
        self._move_to("housing_middle_release_target", [-0.25, -0.3,   -0.03], ensure_neg_z=False)

        self.insert("gearwheel_teeth_35_mod_2_1", "bolt_rotor",
                    gripper_close=0.043, gripper_open=0.05,
            pd={"pre_grasp":  {"position": [0, 0, -0.06]},
                "grasp":       {"position": [0, 0, -0.015]},
                "after_grasp": {"position": [0, 0, -0.25]},
                "pre_asm":     {"position": [0, 0, -0.15]},
                "asm":         {"position": [0, 0, -0.05]},
                "after_asm":   {"position": [0, 0, -0.15]}})

        self.insert("gearwheel_teeth_35_mod_2_2", "bolt_middle_housing",
                    gripper_close=0.043, gripper_open=0.05,
            pd={"pre_grasp":  {"position": [0, 0, -0.06]},
                "grasp":       {"position": [0, 0, -0.015]},
                "after_grasp": {"position": [0, 0, -0.25]},
                "pre_asm":     {"position": [0, 0, -0.15]},
                "asm":         {"position": [0, 0, -0.05]},
                "after_asm":   {"position": [0, 0, -0.15]}})

        self.insert("positioning_pin_d5_20_3", "housing_middle_pin_hole_3",
                    gripper_close=0.0035,
            pd={"pre_grasp":  {"position": [0, 0, -0.02]},
                "grasp":       {"position": [0, 0, -0.0058]},
                "after_grasp": {"position": [0, 0, -0.2]},
                "pre_asm":     {"position": [0, 0, -0.05]},
                "asm":         {"position": [0, 0, -0.0058]},
                "after_asm":   {"position": [0, 0, -0.1]}})

        self.insert("positioning_pin_d5_20_4", "housing_middle_pin_hole_4",
                    gripper_close=0.0035,
            pd={"pre_grasp":  {"position": [0, 0, -0.02]},
                "grasp":       {"position": [0, 0, -0.0058]},
                "after_grasp": {"position": [0, 0, -0.2]},
                "pre_asm":     {"position": [0, 0, -0.05]},
                "asm":         {"position": [0, 0, -0.0058]},
                "after_asm":   {"position": [0, 0, -0.1]}})

        self.insert("tube_nozzle", "housing_top_release_target", gripper_close=0.005,
            pd={"pre_grasp":  {"position": [0, 0, -0.02]},
                "grasp":       {"position": [0, 0, -0.005]},
                "after_grasp": {"position": [0, 0, -0.2]},
                "pre_asm":     {"position": [0, 0, -0.15]},
                "asm":         {"position": [0, 0, -0.0058]},
                "after_asm":   {"position": [0, 0, -0.1]}})

        self.screw("screw_m5_16_hexagon_head_1", "housing_top_screw_hole_1",
                   gripper_close=0.006,
            pd={"pre_grasp":  {"position": [0,    0, -0.02]},
                "grasp":       {"position": [0,    0, -0.004]},
                "after_grasp": {"position": [0,    0, -0.25]},
                "pre_asm":     {"position": [0,    0, -0.05]},
                "asm":         {"position": [0,    0, -0.004]},
                "after_asm":   {"position": [0.05, 0, -0.15]}})

        self.screw("screw_m5_16_hexagon_head_2", "housing_top_screw_hole_2",
                   gripper_close=0.006,
            pd={"pre_grasp":  {"position": [0, 0, -0.02]},
                "grasp":       {"position": [0, 0, -0.004]},
                "after_grasp": {"position": [0, 0, -0.25]},
                "pre_asm":     {"position": [0, 0, -0.05]},
                "asm":         {"position": [0, 0, -0.004]},
                "after_asm":   {"position": [0, 0, -0.15]}})

        self.insert("o_ring_grasp_target", "housing_top", gripper_close=0.0028,
            pd={"pre_grasp":  {"position": [0,       0, -0.03],  "orientation": [0, 0, 0]},
                "grasp":       {"position": [0,       0,  0.003], "orientation": [0, 0, 0]},
                "after_grasp": {"position": [0,       0, -0.25],  "orientation": [0, 0, 0]},
                "pre_asm":     {"position": [-0.0286, 0, -0.03]},
                "asm":         {"position": [-0.0286, 0, -0.01]},
                "after_asm":   {"position": [-0.0286, 0.1, -0.2]}})

        self.insert("positioning_pin_d5_20_5", "housing_top_pin_hole_coverplate_1",
                    gripper_close=0.0035,
            pd={"pre_grasp":  {"position": [0, 0, -0.02]},
                "grasp":       {"position": [0, 0, -0.004]},
                "after_grasp": {"position": [0, 0, -0.2]},
                "pre_asm":     {"position": [0, 0, -0.05]},
                "asm":         {"position": [0, 0, -0.004]},
                "after_asm":   {"position": [0.1, 0.1, -0.2]}})

        self.insert("positioning_pin_d5_20_6", "housing_top_pin_hole_coverplate_2",
                    gripper_close=0.0035,
            pd={"pre_grasp":  {"position": [0, 0, -0.02]},
                "grasp":       {"position": [0, 0, -0.004]},
                "after_grasp": {"position": [0, 0, -0.2]},
                "pre_asm":     {"position": [0, 0, -0.05]},
                "asm":         {"position": [0, 0, -0.004]},
                "after_asm":   {"position": [0.1, 0.1, -0.2]}})

        self.insert("cover_plate", "housing_top", gripper_close=0.014,
            pd={"pre_grasp":  {"position": [0, 0.045, -0.03]},
                "grasp":       {"position": [0, 0.045,  0.004]},
                "after_grasp": {"position": [0, 0.045, -0.35]},
                "pre_asm":     {"position": [0, 0.045, -0.05]},
                "asm":         {"position": [0, 0.045, -0.008]},
                "after_asm":   {"position": [0, 0.045, -0.2]}})

        self.screw("screw_m5_16_hexagon_head_3", "housing_top_screw_hole_coverplate",
                   gripper_close=0.006,
            pd={"pre_grasp":  {"position": [0, 0,   -0.02]},
                "grasp":       {"position": [0, 0,   -0.004]},
                "after_grasp": {"position": [0, 0,   -0.25]},
                "pre_asm":     {"position": [0, 0,   -0.05]},
                "asm":         {"position": [0, 0,   -0.004]},
                "after_asm":   {"position": [0, 0.1, -0.2]}})

        self.insert("tube", "tube_nozzle", gripper_close=0.01,
            pd={"pre_grasp":  {"position": [0, 0, -0.03]},
                "grasp":       {"position": [0, 0,  0.0]},
                "after_grasp": {"position": [0, 0, -0.2]},
                "pre_asm":     {"position": [0, 0, -0.07]},
                "asm":         {"position": [0, 0, -0.03]},
                "after_asm":   {"position": [0, 0, -0.1]}})

        self.clamp("tube_clamp", "tube_nozzle",
                   gripper_close=0.01, gripper_open=0.02, ensure_neg_z=False,
            pd={"pre_grasp":  {"position": [0,    0.04, -0.02]},
                "grasp":       {"position": [0,    0,     0.0]},
                "after_grasp": {"position": [0,    0,    -0.05]},
                "pre_asm":     {"position": [0,    0,    -0.08]},
                "asm":         {"position": [0,    0,    -0.01]},
                "after_asm":   {"position": [-0.1, 0.4,  -0.15]}})

        self.move_to_joint_pos([-0.224, -2.0, 1.78, 1.76, 1.53, 2.92])

        self.insert("housing_assembly_grasp_target", "housing_assembly_release_target",
                    gripper_close=0.03, gripper_open=0.05, ensure_neg_z=False,
            pd={"pre_grasp":  {"position": [-0.01,  0.003, -0.02]},
                "grasp":       {"position": [-0.01,  0.003,  0.024]},
                "after_grasp": {"position": [-0.20,  0.003,  0.024]},
                "pre_asm":     {"position": [ 0.2,   0.003,  0.024]},
                "asm":         {"position": [ 0.0,   0.003,  0.024]},
                "after_asm":   {"position": [ 0.0,   0.003, -0.1]}})

        self.screw("screw_m5_16_hexagon_head_4", "housing_bottom_screw_hole_1",
                   gripper_close=0.006,
            pd={"pre_grasp":  {"position": [0, 0,   -0.02]},
                "grasp":       {"position": [0, 0,   -0.004]},
                "after_grasp": {"position": [0, 0,   -0.25]},
                "pre_asm":     {"position": [0, 0,   -0.05]},
                "asm":         {"position": [0, 0,   -0.004]},
                "after_asm":   {"position": [0, 0,   -0.1]}})

        self.screw("screw_m5_16_hexagon_head_5", "housing_bottom_screw_hole_2",
                   gripper_close=0.006,
            pd={"pre_grasp":  {"position": [0, 0,   -0.02]},
                "grasp":       {"position": [0, 0,   -0.004]},
                "after_grasp": {"position": [0, 0,   -0.25]},
                "pre_asm":     {"position": [0, 0,   -0.05]},
                "asm":         {"position": [0, 0,   -0.004]},
                "after_asm":   {"position": [0, 0.2, -0.05]}})

        print(f"Total time: {time.time() - t0:.1f}s")

    # ── entry points ───────────────────────────────────────────────────────────

    def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        for _ in range(100):   # physics warm-up
            self.scene.step()
        self.labit_policy()

    def _signal_handler(self, *_):
        print("\n[EXIT] Interrupted.")
        os._exit(0)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--viewer", action="store_true", help="Enable viewer (requires display)")
    args = p.parse_args()
    sim = LabItGenesis(headless=not args.viewer)
    sim.run()
