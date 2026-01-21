"""
Position based insertion in development mode and use for following tests:
 - Collect the training data for insertion-net
 - Randomize the pyhsical parameter in MuJoCo to compare the force distribution with the real robot
 - Compare the mesh decomposition with different mesh scale
 - Test the surface toughness with the sphere-based method
"""
# for recording videos headless in mujoco
import os
import sys
import signal
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import mujoco.viewer

import time
import imageio

from qbit.utils.tf_utils import T
from qbit.utils.mj_viewer_utils import update_view_camera_parameter
from qbit.utils.mujoco_utils import get_relative_pose, convert_quat_to_wxyz
from qbit.utils.data_recording_utils import DataRecording
from qbit.sim_envs.mujoco_env_insertion import MujocoEnvBase
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

NUM_RUNS = 300

RESULT_DIR = "/workspace/examples/experiment_results/position_based/exp_labit_benchmark"

SIM_TIMESTEP = 0.0005 #0.0005, 0.001

NN_CONTROL_DT = 0.01
ADMITTANCE_CONTROL_T = 0.01
JOINT_POSITION_CONTROL_T = 0.001  # Second

POS_RANDOM_LIMIT = 0.001 # meter
ROT_RANDOM_LIMIT = 1.5 # degree


class PositionBasedInsertion(MujocoEnvBase):
    
    
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
        
        self.data_recording = DataRecording(task_env_config_path=task_env_config_path,
                                            robot=self.robot,
                                            sim_timestep=sim_timestep,
                                            live_plotting=False)

       
        self._mj_renderer = mujoco.Renderer(self._mj_model, height=720, width=1280)
        self.cam = mujoco.MjvCamera()

        # Example: set camera position and orientation
        self.cam.azimuth = 0       # horizontal angle
        self.cam.elevation = -60    # vertical angle
        self.cam.distance = 1.0     # distance to model center
        self.cam.lookat = [-0.2, 0, 1] # center point
        self.frames = []
        self.fps = 24
        self.iterations_per_frame = int(1/self._sim_timestep/self.fps)


    def termination(self, 
                    pose_goal,
                    pose_current,
                    threshold: float = 0.001
                    ) -> bool:

        # Position error.
        error_pos = pose_goal.translation - pose_current.translation

        # Orientation error.
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        error_ori = np.zeros(3)
        site_quat = pose_current.quaternion
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, pose_goal.quaternion, site_quat_conj)
        mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
                
        if (np.linalg.norm(error_pos) < 0.0001 and np.linalg.norm(error_ori) < 0.001) or (all(np.abs(self._mj_data.qvel[0:6]) < 0.00001)):
            return True

        return False
    
    
    def minjerk_s(self, t, T):
        tau = np.clip(t/T, 0.0, 1.0)
        return tau #10*tau**3 - 15*tau**4 + 6*tau**5


    def minimal_jerk_pose(self, p0, q0, p1, q1, t, T):
        # scalar-last order – (x, y, z, w)
        s = self.minjerk_s(t, T)
        p_t = p0 + s * (p1 - p0)
        # q_t = geometric_slerp(q0, q1, s)
        slerp = Slerp(times=[0.0, 1.0], rotations=R.concatenate([R.from_quat(q0), R.from_quat(q1)]))
        q_t = slerp(s)
        return p_t, q_t.as_quat()


    def move_pose_lin(self,
                  viewer,
                  body_name: str,
                  _goal_pose_T: T = None,
                  pos_offset: np.array = np.array([0.0, 0.0, 0.0]),
                  quat_offset: np.array = np.array([0.0, 0.0, 0.0, 1.0]),   # xyzw
                  label: str = "moving",
                  ensure_negative_z_axis=True
                  ):
        # quaternion = R.from_quat(_goal_pose_T.quaternion) * R.from_quat(quat_offset)
        # quaternion = quaternion.as_quat()
        quaternion = quat_offset
        translation = _goal_pose_T.translation + pos_offset
        _goal_pose_T = T(translation=translation,
                         quaternion=quaternion)
        current_eef_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", "tool0", ensure_negative_z_axis=False)

        self.show_mj_target_frame(body_name, _goal_pose_T, ensure_negative_z_axis=ensure_negative_z_axis)

        q_init = self.robot.get_current_joint_state()[0]
        q_goal = self.robot._eef_position_controller.ik.ik(_goal_pose_T._matrix, q_init)

        i = 0
        dt = self._sim_timestep
        max_eef_vel = 1.0       # m/s
        max_joint_vel = 3.33    # rad/s
        traj_time = 1.0 #np.linalg.norm(_goal_pose_T.translation - current_eef_pose_T.translation) / max_eef_vel
        nsteps_new_qtgoal = int((traj_time//dt)//10)

        with np.printoptions(precision=4, floatmode="fixed", suppress=True):
            print("[ROBOT] moving to target pose p: {}, q: {}".format(_goal_pose_T.translation, _goal_pose_T.quaternion))
            
        while True:
            step_start = time.time()
            t = i*self._sim_timestep
            
            if i % nsteps_new_qtgoal == 0:
                print("> [ROBOT] moving ... t: {:.4f}".format(t))
            q_current = self.robot.get_current_joint_state()[0]
            pt, qt = self.minimal_jerk_pose(p0=current_eef_pose_T.translation,
                                            q0=current_eef_pose_T.quaternion,
                                            p1=_goal_pose_T.translation,
                                            q1=_goal_pose_T.quaternion,
                                            t=t,
                                            T=traj_time)
            qt_goal = self.robot._eef_position_controller.ik.ik(T(pt, qt)._matrix, q_current)
            self._mj_data.ctrl[0:6] = qt_goal

            for _ in range(2):
                self.step_mj_simulation()

            self.data_recording.record(label=label)
            self._mj_renderer.update_scene(self._mj_data, camera=self.cam)

            if i % self.iterations_per_frame == 0.0:
                frame = self._mj_renderer.render()
                self.frames.append(frame)

            if viewer != None:    
                viewer.sync()

            _eef_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", "tool0", ensure_negative_z_axis=False)
            if self.termination(pose_goal=_goal_pose_T, pose_current=_eef_pose_T, threshold=0.002) or t >= 2.0:
                break

            i += 1

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        self._mj_data.qvel[0:6] = 0.0
        
        with np.printoptions(precision=4, floatmode="fixed", suppress=True):
            print("> [ROBOT] reached target pose p: {}, q: {}".format(_eef_pose_T.translation, _eef_pose_T.quaternion))


    def show_mj_target_frame(self, body_name, _goal_pose_T, ensure_negative_z_axis=True):
        base_pose = get_relative_pose(self._mj_model, self._mj_data, "world", "base", ensure_negative_z_axis=False)
        mocap_id = self._mj_model.body("target").mocapid[0]
        eef_offset = np.array([0.0, 0.0, 0.21])

        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in the MuJoCo model")
        
        # Get body rotation matrix (world-from-body)
        body_R = self._mj_data.xmat[body_id].reshape(3, 3)
        if body_R[2, 2] > 0 and ensure_negative_z_axis:
            body_R = body_R @ np.diag([1.0, -1.0, -1.0])

        # Transform to world frame
        eef_offset = body_R @ eef_offset 
        
        translation = _goal_pose_T.translation
        self._mj_data.mocap_pos[mocap_id, :] = base_pose.matrix[:3,:3] @ translation + base_pose.translation + eef_offset
        R_ = base_pose.matrix[:3,:3] @ _goal_pose_T._matrix[:3,:3]
        quat = R.from_matrix(R_).as_quat()  # xyzw

        self._mj_data.mocap_quat[mocap_id, :] = convert_quat_to_wxyz(quat)  # wxyz
    

    def set_gripper_position(self, position: float, viewer):
        """
        Set the gripper position (width between fingers).
        position: float, range from [0.0, 0.05] meter [fully closed, fully open]
        """
        position = np.clip(position, 0, 0.05)
        i = 0
        print("> [GRIPPER] setting gripper opening to {:.4f}.".format(position))
        while True:
            t = i*self._sim_timestep

            self._mj_data.ctrl[6] = 0.025 - position/2

            mujoco.mj_step(self._mj_model, self._mj_data)
      
            position_error = np.abs(self._mj_data.qpos[6] - (0.025 - position/2))
            
            self.data_recording.record(label="grasping")
            self._mj_renderer.update_scene(self._mj_data, camera=self.cam)

            if i % self.iterations_per_frame == 0.0:
                frame = self._mj_renderer.render()
                self.frames.append(frame)
            
            if viewer is not None:
                viewer.sync()
            
            if position_error <= 0.0001 or (np.abs(self._mj_data.qvel[6]) <= 0.0001 and np.abs(self._mj_data.qvel[7]) <= 0.0001):
                break
            
            i += 1
        print("> [GRIPPER] done.")


    def get_offset_in_body_frame(self, body_name: str, pos_offset: np.array = np.array([0.0,0.0,0.0]), euler_offset: np.array = np.array([0.0,0.0,0.0]), ensure_negative_z_axis = True):
        """
        Compute a position and orientation offset expressed in the robot "base" body frame
        given offsets specified in a named MuJoCo body frame.
        This method:
        - Looks up the MuJoCo body by name and reads the body's world pose (rotation matrix and position)
            from self._mj_data.
        - Applies an internal tool/end-effector offset (a fixed 0.228 m displacement along the base z-axis)
            that is first expressed in the body frame and then added to the provided position offset.
        - Transforms the combined position offset from the specified body frame into the base frame.
        - Converts the provided Euler-angle offset (XYZ order, radians) into a rotation matrix,
            applies the body->world and world->base transforms, and returns the result as a MuJoCo-style
            quaternion (w, x, y, z).
        Parameters
        ----------
        body_name : str
                Name of the MuJoCo body whose local frame the input offsets are specified in.
        pos_offset : numpy.ndarray, shape (3,), optional
                Cartesian position offset expressed in the named body frame (in meters).
                Default: np.array([0.0, 0.0, 0.0]).
        euler_offset : numpy.ndarray, shape (3,), optional
                Euler-angle orientation offset expressed in the named body frame (in degrees).
                Angles are applied in 'xyz' order (i.e., rotate about x, then y, then z).
                Default: np.array([0.0, 0.0, 0.0]).
        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
                - pos_offset_base : numpy.ndarray, shape (3,)
                        The position offset expressed in the "base" body frame (meters).
                        This is computed by: base_R^T * ( body_R * (pos_offset + tool_offset_in_body) ),
                        where body_R is the world-from-body rotation and base_R is the world-from-base rotation.
                - quat_offset_base : numpy.ndarray, shape (4,)
                        The orientation offset expressed in the "base" body frame as a MuJoCo quaternion
                        in the order (w, x, y, z). The quaternion corresponds to the rotation that, when
                        applied in the base frame, produces the same orientation offset as the input Euler
                        angles expressed in the body frame.
        Raises
        ------
        ValueError
                If the named body (body_name) is not found in the MuJoCo model.
        IndexError
                If the "base" body is not present in the MuJoCo model or internal MuJoCo data arrays
                cannot be indexed as expected (the code currently assumes a body named "base" exists).
        TypeError
                If pos_offset or euler_offset cannot be interpreted as 3-element numeric arrays.
        """
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, body_name)

        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in the MuJoCo model")

        # Get body rotation matrix (world-from-body)
        body_R = self._mj_data.xmat[body_id].reshape(3, 3)

        # Ensure the body-frame Z axis (third column of body_R) points downwards
        # If the Z axis has a positive world-Z component, rotate 180deg about the
        # body-local X axis (diag([1,-1,-1])) so Z becomes negative while keeping a proper rotation.
        if body_R[2, 2] > 0 and ensure_negative_z_axis:
            body_R = body_R @ np.diag([1.0, -1.0, -1.0])

        body_pos = self._mj_data.xpos[body_id]

        base_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base")
        base_R = self._mj_data.xmat[base_id].reshape(3, 3)
        base_pos = self._mj_data.xpos[base_id]

        # Transform position offset from body frame to world frame:
        # body_R is world-from-body rotation
        if body_name == "base": eef_offset = np.array([0.0, 0.0, 0.21])#np.array([0.0, 0.0, 0.228])
        else: eef_offset = np.array([0.0, 0.0, -0.21]) #np.array([0.0, 0.0, -0.228])

        pos_offset_world = body_R @ (pos_offset + eef_offset) # regard for offset from tool0 to grasping fingers

        # Convert euler offset (assumed in degrees, order XYZ) to rotation matrix in world frame
        rot_offset_body = R.from_euler('xyz', euler_offset, degrees=True).as_matrix()
        rot_offset_world = body_R @ rot_offset_body

        # Convert rotation matrix to MuJoCo quaternion (w, x, y, z)
        quat_offset_world = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat_offset_world, rot_offset_world.reshape(9))

        # Transform from world frame to base frame using base transformation
        base_R_inv = base_R.T  # Inverse of rotation matrix is its transpose
        pos_offset_base = base_R_inv @ pos_offset_world
        rot_offset_base = base_R_inv @ rot_offset_world

        # Convert to quaternion in base frame
        quat_offset_base = R.from_matrix(rot_offset_base).as_quat()

        return pos_offset_base, quat_offset_base
    

    def insert(self, viewer, body_name: str, target_name: str, poses_dict: dict, ensure_negative_z_axis: bool = True, gripper_opening: float = 0.017, gripper_closing: float = 0.0):
        print("[INSERTING] {} into {}.".format(body_name, target_name))
        # set gripper opening
        self.set_gripper_position(gripper_opening, viewer)

        # move above body
        # on real robot, object pose would be calibrated and saved in a yaml file.
        # on real robot, we cant plan grasp pose relative to object pose, because we dont implement pose detection.
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", body_name, ensure_negative_z_axis=ensure_negative_z_axis) 
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=body_name, pos_offset=poses_dict["pre_grasp"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["pre_grasp"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=body_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)

        # move to grasp pose (body coordinate frame in between fingertips)
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", body_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=body_name, pos_offset=poses_dict["grasp"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["grasp"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=body_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)

        # close the gripper to grasp
        self.set_gripper_position(gripper_closing, viewer)

        # move away
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", body_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=body_name, pos_offset=poses_dict["after_grasp"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["after_grasp"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=body_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)

        # move above assembly target
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", target_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=target_name, pos_offset=poses_dict["pre_asm"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["pre_asm"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=target_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)

        # correct orientation for insertion; arising from object in hand movement/slipping/unstable grasps -> pose uncertainty
        # correction_quat = self.pose_correction(viewer, _goal_pose_T=goal_pose_T, body_name=body_name, target_name=target_name, ensure_negative_z_axis=ensure_negative_z_axis)

        # assemble body and target
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", target_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=target_name, pos_offset=poses_dict["asm"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["asm"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=target_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="inserting", ensure_negative_z_axis=ensure_negative_z_axis)

        # release
        self.set_gripper_position(gripper_opening, viewer)

        # move away
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", target_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=target_name, pos_offset=poses_dict["after_asm"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["after_asm"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=target_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)
    

    def screw(self, viewer, body_name: str, target_name: str, poses_dict: dict, ensure_negative_z_axis: bool = True, gripper_opening: float = 0.017, gripper_closing: float = 0.0):
        print("[SCREWING] {} into {}.".format(body_name, target_name))
        # set gripper opening
        self.set_gripper_position(gripper_opening, viewer)

        # move above body
        # on real robot, object pose would be calibrated and saved in a yaml file.
        # on real robot, we cant plan grasp pose relative to object pose, because we dont implement pose detection.
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", body_name, ensure_negative_z_axis=ensure_negative_z_axis) 
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=body_name, pos_offset=poses_dict["pre_grasp"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["pre_grasp"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=body_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)

        # move to grasp pose (body coordinate frame in between fingertips)
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", body_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=body_name, pos_offset=poses_dict["grasp"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["grasp"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=body_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)

        # close the gripper to grasp
        self.set_gripper_position(gripper_closing, viewer)

        # move away
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", body_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=body_name, pos_offset=poses_dict["after_grasp"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["after_grasp"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=body_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)

        # move above assembly target
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", target_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=target_name, pos_offset=poses_dict["pre_asm"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["pre_asm"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=target_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)

        ### screw assembly skill
        # assemble body and target
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", target_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=target_name, pos_offset=poses_dict["asm"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["asm"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=target_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="inserting", ensure_negative_z_axis=ensure_negative_z_axis)

        ## regrasp
        self.set_gripper_position(0.01, viewer)
        # move to grasp pose (body coordinate frame in between fingertips)
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", body_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=body_name, pos_offset=poses_dict["grasp"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["grasp"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=body_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)

        self.set_gripper_position(gripper_closing, viewer)

        # perform screwing motion - rotate around z-axis
        num_screw_rotations = 2
        rotation_angle_per_step = np.pi  # radians per simulation step
        current_pose = self.robot.get_eef_pose_in_base_frame()

        for rotation_step in range(int(num_screw_rotations * (2 * np.pi) / rotation_angle_per_step)):
            print("[SCREWING] ROTATION {}.".format(rotation_step))
            # Calculate rotation increment around z-axis
            rotation_increment = R.from_euler('z', rotation_angle_per_step).as_matrix()
            
            # Apply rotation to current orientation
            current_rot = R.from_quat(current_pose.quaternion).as_matrix()
            new_rot = current_rot @ rotation_increment
            new_quat = R.from_matrix(new_rot).as_quat()
            
            self.move_pose_lin(viewer=viewer, body_name=target_name, _goal_pose_T=current_pose, quat_offset=new_quat, label="screwing", ensure_negative_z_axis=ensure_negative_z_axis)
            
            # Update current pose for next iteration
            current_pose = self.robot.get_eef_pose_in_base_frame()

        # release
        self.set_gripper_position(gripper_opening, viewer)

        # move away
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", target_name, ensure_negative_z_axis=ensure_negative_z_axis)
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name=target_name, pos_offset=poses_dict["after_asm"].get("position", np.array([0.0, 0.0, 0.0])), euler_offset=poses_dict["after_asm"].get("orientation", np.array([0.0, 0.0, 0.0])), ensure_negative_z_axis=ensure_negative_z_axis)
        self.move_pose_lin(viewer=viewer, body_name=target_name, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", ensure_negative_z_axis=ensure_negative_z_axis)


    def labit_policy(self, viewer = None):
        # # in real robot experiment, calibrate following poses:
        # plate_benchmark: pose relative to robot base
        # housing_middle: pose relative to plate_benchmark
        # housing_bottom: pose relative to plate_benchmark
        # housing_top: pose relative to plate_benchmark 
        # for each object:
        #   (relative to base): object pose
        #   (object centric): pre_grasp, grasp, after_grasp, 
        #   (target centric): pre_asm, asm, after_asm
        # #
       
        # assembly of housing middle components
        # self.insert(viewer=viewer, body_name="pcb_body", target_name="housing_middle_pcb_target_body", gripper_closing=0.006,
        #     poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.03])},
        #                 "grasp": {"position": np.array([0.0, 0.0, -0.005])},
        #                 "after_grasp": {"position": np.array([0.0, 0.0, -0.25])},
        #                 "pre_asm": {"position": np.array([0.0, 0.0, -0.1])},
        #                 "asm": {"position": np.array([0.0, 0.0, -0.015])},
        #                 "after_asm": {"position": np.array([0.0, 0.05, -0.11])}})
        # self.insert(viewer=viewer, body_name="plug_inside_loose_1_body", target_name="plug_inside_fixed_1_body", gripper_closing=0.005,
        #     poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.03])},
        #                 "grasp": {"position": np.array([0.0, 0.0, -0.005])},
        #                 "after_grasp": {"position": np.array([0.0, 0.0, -0.25])},
        #                 "pre_asm": {"position": np.array([0.0, 0.0, -0.1])},
        #                 "asm": {"position": np.array([0.0, 0.0, -0.009])},
        #                 "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        # self.insert(viewer=viewer, body_name="plug_inside_loose_2_body", target_name="plug_inside_fixed_2_body", gripper_closing=0.005,
        #     poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.03])},
        #                 "grasp": {"position": np.array([0.0, 0.0, -0.005])},
        #                 "after_grasp": {"position": np.array([0.0, 0.0, -0.25])},
        #                 "pre_asm": {"position": np.array([0.0, 0.0, -0.1])},
        #                 "asm": {"position": np.array([0.0, 0.0, -0.009])},
        #                 "after_asm": {"position": np.array([0.1, 0.0, -0.1])}})
        # self.insert(viewer=viewer, body_name="plug_outside_loose_body", target_name="plug_outside_fixed_body", gripper_closing=0.009,
        #     poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.03])},
        #                 "grasp": {"position": np.array([0.0, 0.0, -0.001])},
        #                 "after_grasp": {"position": np.array([0.0, 0.0, -0.25])},
        #                 "pre_asm": {"position": np.array([0.0, 0.0, -0.1])},
        #                 "asm": {"position": np.array([0.0, 0.0, -0.001])},
        #                 "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})

        # # assembly for housing bottom components
        self.insert(viewer=viewer, body_name="positioning_pin_d5_20_2_body",target_name="housing_bottom_pin_hole_2_body", gripper_closing=0.0035,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        self.insert(viewer=viewer, body_name="positioning_pin_d5_20_1_body", target_name="housing_bottom_pin_hole_1_body", gripper_closing=0.0035,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp":  {"position": np.array([0.0, 0.0, -0.004])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        self.insert(viewer=viewer, body_name="bolt_rotor_body", target_name="housing_bottom_body", gripper_closing=0.005,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.04])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.0058])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.15])},
                        "asm": {"position": np.array([0.0, 0.0, -0.06])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.15])}})

        # assemble housing middle onto housing bottom
        self.insert(viewer=viewer, body_name="housing_middle_grasp_target_body", target_name="housing_middle_release_target_body", ensure_negative_z_axis=False,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.003, -0.06])},
                        "grasp": {"position": np.array([0.0, 0.003, 0.024])},
                        "after_grasp": {"position": np.array([0.15, 0.003, 0.024])},
                        "pre_asm": {"position": np.array([-0.15, 0.003, 0.024])},
                        "asm": {"position": np.array([-0.001, 0.003, 0.024])},
                        "after_asm": {"position": np.array([-0.001, 0.003, -0.03])}})
        # move to side to avoid collision with housing bottom + middle
        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", "housing_middle_release_target_body", ensure_negative_z_axis=False) 
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name="housing_middle_release_target_body", pos_offset=np.array([-0.25, 0.003, -0.03]), ensure_negative_z_axis=False)
        self.move_pose_lin(viewer=viewer, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", body_name="housing_middle_release_target_body")

        goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", "housing_middle_release_target_body", ensure_negative_z_axis=False) 
        pos_offset, quat_offset = self.get_offset_in_body_frame(body_name="housing_middle_release_target_body", pos_offset=np.array([-0.25, -0.3, -0.03]), ensure_negative_z_axis=False)
        self.move_pose_lin(viewer=viewer, _goal_pose_T=goal_pose_T, pos_offset=pos_offset, quat_offset=quat_offset, label="moving", body_name="housing_middle_release_target_body")


        # place gearwheel 1
        self.insert(viewer=viewer, body_name="gearwheel_teeth_35_mod_2_1_body", target_name="bolt_rotor_body", gripper_closing=0.043, gripper_opening=0.05,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.06])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.015])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.25])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.15])},
                        "asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.15])}})
        # place gearwheel 2
        self.insert(viewer=viewer, body_name="gearwheel_teeth_35_mod_2_2_body", target_name="bolt_middle_housing_body", gripper_closing=0.043, gripper_opening=0.05,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.06])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.015])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.25])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.15])},
                        "asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.15])}})
        # insert pin into housing middle
        self.insert(viewer=viewer, body_name="positioning_pin_d5_20_3_body", target_name="housing_middle_pin_hole_3_body", gripper_closing=0.0035,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.0058])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.0058])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        # insert pin into housing middle
        self.insert(viewer=viewer, body_name="positioning_pin_d5_20_4_body", target_name="housing_middle_pin_hole_4_body", gripper_closing=0.0035,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.0058])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.0058])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        # put housing top onto housing middle
        self.insert(viewer=viewer, body_name="tube_nozzle_body", target_name="housing_top_release_target_body", gripper_closing=0.005,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position":np.array([0.0, 0.0, -0.005])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.15])},
                        "asm": {"position": np.array([0.0, 0.0, -0.0058])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        # screw 1
        self.screw(viewer=viewer, body_name="screw_m5_16_hexagon_head_1_body", target_name="housing_top_screw_hole_1_body", gripper_closing=0.006,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.25])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_asm": {"position": np.array([0.05, 0.0, -0.15])}})
        # screw 2
        self.insert(viewer=viewer, body_name="screw_m5_16_hexagon_head_2_body", target_name="housing_top_screw_hole_2_body", gripper_closing=0.006,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        # o-ring
        self.insert(viewer=viewer, body_name="o_ring_grasp_target_body", target_name="housing_top_body", gripper_closing=0.0028,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.03]), "orientation": np.array([0.0, 0.0, 0.0])},
                        "grasp": {"position": np.array([0.0, 0.0, 0.003]), "orientation": np.array([0.0, 0.0, 0.0])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.25]), "orientation": np.array([0.0, 0.0, 0.0])},
                        "pre_asm": {"position": np.array([-0.0286, 0.0, -0.03])},
                        "asm": {"position": np.array([-0.0286, 0.0, -0.015])},
                        "after_asm": {"position": np.array([-0.0286, 0.1, -0.2])}})

        # pin for coverplate 1
        self.insert(viewer=viewer, body_name="positioning_pin_d5_20_5_body", target_name="housing_top_pin_hole_coverplate_1_body", gripper_closing=0.01,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_asm": {"position": np.array([0.0, 0.1, -0.2])}})
        # pin for coverplate 2
        self.insert(viewer=viewer, body_name="positioning_pin_d5_20_6_body", target_name="housing_top_pin_hole_coverplate_2_body", gripper_closing=0.01,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.15])}})
        # coverplate
        self.insert(viewer=viewer, body_name="cover_plate_body", target_name="housing_top_body", gripper_closing=0.01,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.04, -0.03])},
                        "grasp": {"position": np.array([0.0, 0.04, -0.01])},
                        "after_grasp": {"position": np.array([0.0, 0.04, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.04, -0.05])},
                        "asm": {"position": np.array([0.0, 0.04, -0.01])},
                        "after_asm": {"position": np.array([0.0, 0.04, -0.1])}})

        # screw 3 for coverplate
        self.insert(viewer=viewer, body_name="screw_m5_16_hexagon_head_3_body", target_name="housing_top_screw_hole_coverplate_body", gripper_closing=0.006,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_asm": {"position": np.array([0.0, 0.1, -0.2])}})
        # tube
        self.insert(viewer=viewer, body_name="tube_body", target_name="tube_nozzle_body", gripper_closing=0.01,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.03])},
                        "grasp": {"position": np.array([0.0, 0.0, 0.0])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.07])},
                        "asm": {"position": np.array([0.0, 0.0, -0.03])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})

        # tube clamp
        self.insert(viewer=viewer, body_name="tube_clamp_body", target_name="tube_nozzle_body", gripper_closing=0.01, gripper_opening=0.02, ensure_negative_z_axis=True,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, 0.0])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.05])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.08])},
                        "asm": {"position": np.array([0.0, 0.0, -0.01])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        
        # rotate assembly and put onto "housing middle" starting point
        self.insert(viewer=viewer, body_name="housing_assembly_grasp_target_body", target_name="housing_assembly_release_target_body", gripper_closing=0.01, gripper_opening=0.03, ensure_negative_z_axis=False,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.003, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.003, 0.024])},
                        "after_grasp": {"position": np.array([-0.20, 0.003, 0.024])},
                        "pre_asm": {"position": np.array([0.05, 0.003, 0.024])},
                        "asm": {"position": np.array([0.0, 0.003, 0.024])},
                        "after_asm": {"position": np.array([0.0, 0.003, -0.1])}})
        # screw 4
        self.insert(viewer=viewer, body_name="screw_m5_16_hexagon_head_4_body", target_name="housing_bottom_screw_hole_1_body", gripper_closing=0.006,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        # screw 5
        self.insert(viewer=viewer, body_name="screw_m5_16_hexagon_head_5_body", target_name="housing_bottom_screw_hole_2_body", gripper_closing=0.006,
            poses_dict={"pre_grasp": {"position": np.array([0.0, 0.0, -0.02])},
                        "grasp": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_grasp": {"position": np.array([0.0, 0.0, -0.2])},
                        "pre_asm": {"position": np.array([0.0, 0.0, -0.05])},
                        "asm": {"position": np.array([0.0, 0.0, -0.004])},
                        "after_asm": {"position": np.array([0.0, 0.0, -0.1])}})
        
        self.data_recording.save()
        self.data_recording.plot_data()

        return


    def exec_labit(self):
        """
        Main function to execute the LABIT benchmark task.
        """
        signal.signal(signal.SIGINT, self.signal_handler)
        # self.apply_gravity_compensation() # wont work; did it in xml
        with mujoco.viewer.launch_passive(self._mj_model, self._mj_data, show_left_ui=False, show_right_ui=False) as viewer:
            self.update_view_scale()
            self.update_view_opt(viewer)
            update_view_camera_parameter(viewer, view_type="labit_benchmark")
            
            for _ in range(10):
                self.step_mj_simulation()
            viewer.sync()

            self.labit_policy(viewer=viewer)
            
            imageio.mimsave("output.mp4", self.frames, fps=self.fps)
            viewer.close()


    def signal_handler(self, sig, frame):
        print("\n[EXIT]benchmark execution got interrupted. Saving video until current timestamp.")
        try:
            imageio.mimsave("output.mp4", self.frames, fps=self.fps)
        except Exception as e:
            print(f"Error saving video: {e}")
        
        print("\n[EXIT]saving recorded data.")
        try:
            self.data_recording.save()
            self.data_recording.plot_data()
        except Exception as e:
            print(f"Error saving/plotting data: {e}")

        os._exit(0) 


    def exec_labit_headless(self):
        signal.signal(signal.SIGINT, self.signal_handler)

        self.labit_policy()

        imageio.mimsave("output.mp4", self.frames, fps=self.fps)
        
        self.data_recording.save()
        self.data_recording.plot_data()
        
        sys.exit(0)

  
if __name__ == "__main__":
    
    task_env_config_path = "/workspace/qbit/configs/envs/ur5e_labit_benchmark.yaml"
    
    mj = PositionBasedInsertion(
        task_env_config_path=task_env_config_path,
        server_modus=True,
        sim_timestep=SIM_TIMESTEP,
        )
    mj.exec_labit()
    # mj.exec_labit_headless()
