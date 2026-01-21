"""
UR5e robot class
"""
from typing import List
import copy
import numpy as np
from tracikpy import TracIKSolver

import mujoco

from qbit.utils.tf_utils import T
from qbit.utils.mujoco_utils import convert_quat_to_xyzw, get_relative_pose
from qbit.robots.robot_base import RobotBase

class UR5eMjArm(RobotBase):
    
    
    def __init__(self,
                 qbit_robot_config: dict,
                 headless: bool = False,
                 arm_base_pos = (0.2, 0, 0.95),
                 arm_base_qua = (0, 0, 0, -1), # wxyz # TODO: Use the default base quaternion in xml, check and change this later
                 ik_solver = 'trac_ik',
                 ) -> None:
        
        super().__init__(qbit_robot_config,
                         headless,
                         arm_base_pos,
                         arm_base_qua)
        
        self.load_ik_solver()

        # TODO: parse using xml file
        self.base_link_name = 'base'
        self.base_id = 1
        self.tcp_body_name = 'tool0'
        self.tcp_id = 8

    def get_q_goal(self, eef_pose):
        eef_pose_T = T(
            translation=eef_pose[:3],
            quaternion=eef_pose[3:]
        )
        q_current, _ = self.get_current_joint_state()
        q_goal = self.ik_solver.ik(eef_pose_T.matrix, q_current)
        return q_goal

    def move_to_eef_pose(self,
                         viewer: None,
                         eef_pose: List[float],
                         qpos_thresh: float = 0.01,
                         executing=False,
                         ):
        """
        Move the ee to the desired pose
        Using the ik solver to compute the joint angles
        """
        # print("Move to eef pose:", eef_pose)
        eef_pose_T = T(
            translation=eef_pose[:3],
            quaternion=eef_pose[3:]
        )
        q_current, _ = self.get_current_joint_state()
        q_goal = self.ik_solver.ik(eef_pose_T.matrix, q_current)
        self._goal_joint_pos = q_goal

        # print(eef_pose[:3])
        # print("Q current: ", q_current)
        # print("Goal joint pos: ", q_goal)

        if executing:
            qpos_err = np.abs(self._mj_data.qpos[:6] - q_goal)
            i = 0
            while (qpos_err > qpos_thresh).any():
                self.spin()
                mujoco.mj_step(self._mj_model, self._mj_data)
                qpos_err = np.abs(self._mj_data.qpos[:6] - q_goal)
                i += 1
                                
                if viewer: viewer.sync()
            print(f"[MOVE TO EEF POSE] Reached the goal in {i} steps with ERROR: {qpos_err}")
            return q_goal
        return
    
    def spin(self):
        """
        Run low-level position control loop
        """
        q_pos_curr, q_vel_curr = self.get_current_joint_state()
        q_cmd = self._joint_position_controller.pd_joint_position_control(
            q_pos_curr,
            self._goal_joint_pos,
            q_vel_curr)
        # forwards the joint command to the mujoco ctrl
        # print("Joint command: ", q_cmd)
        
        self._mj_data.ctrl[0:6] = q_cmd + q_pos_curr
        # self._mj_data.qpos[0:6] =  q_cmd + q_pos_curr

        # print(q_cmd)
        return

    def load_ik_solver(self):
        
        self._ik = TracIKSolver(
            "/workspace/qbit/assets/robots/ur5e/ur5e_robot_calibrated.urdf",
            base_link="base_link",
            tip_link="tool0",
            timeout=0.5,
            epsilon=1e-6,
            solve_type='Distance'
        )
        return

    @property
    def ik_solver(self):
        return self._ik


    def check_fk_error(self):
        """
        Check the FK error
        """

        tcp_in_base_T = self.get_eef_pose_in_base_frame()

        # IK
        q_pos = self._mj_data.qpos
        eef_fk_out = self.ik_solver.fk(q_pos)
        
        # visualize the TCP pose calculated by FK with mocap site
        tcp_fk_T = self._base_T * T.from_matrix(eef_fk_out)
        # self._mj_data.mocap_pos[0] = tcp_fk_T.translation
        # self._mj_data.mocap_quat[0] = convert_quat_to_wxyz(tcp_fk_T.quaternion)
        
        # calculate the differences
        eef_diff = np.linalg.inv(tcp_in_base_T.matrix) @ eef_fk_out
        trans_err = np.linalg.norm(eef_diff[:3, 3], ord=1)
        angle_err = np.arccos((np.trace(eef_diff[:3, :3]) - 1) /2)
        
        print("*" * 50)
        print(f"TCP pose in base from Mujoco: {tcp_in_base_T}")
        print(f"TCP pose from FK: {eef_fk_out}")
        print(f"[FK Error] tran: {trans_err} // angle: {angle_err}")
        return
    
    def get_eef_pose_in_base_frame(self) -> T:
        """
        Get the EEF pose in the base frame
        It is the ground truth pose!!!
        """
        # # TCP in world
        # tcp_in_world_pos = self._mj_data.xpos[self.tcp_id, :]
        # tcp_in_world_qua = self._mj_data.xquat[self.tcp_id, :]
        # self._tcp_T = T(
        #     translation = tcp_in_world_pos,
        #     quaternion = convert_quat_to_xyzw(tcp_in_world_qua)
        # )
        # # Base in world
        # base_in_world_pos = self._mj_data.xpos[self.base_id, :]
        # base_in_world_quat = self._mj_data.xquat[self.base_id, :]
        # self._base_T = T(
        #     translation = base_in_world_pos,
        #     quaternion = convert_quat_to_xyzw(base_in_world_quat)
        # )
        
        # tcp_in_base_T = np.linalg.inv(self._base_T.matrix) @ self._tcp_T.matrix

        return get_relative_pose(self._mj_model, self._mj_data, self.base_link_name, self.tcp_body_name, ensure_negative_z_axis=False)
    

    def get_peg_pos_quat(self):
        return np.array(self._mj_data.xpos[9, :]) ,  np.array(self._mj_data.xquat[9, :])

    def get_fts_data(self,
                     transform_to_base: bool = False):
        """
        Get the FTS data
        If transform_to_base is True, the FTS is transformed to use the base frame coordinate
        """
        sensor_data_corrected = - copy.deepcopy(self._mj_data.sensordata)
        if not transform_to_base:
            return sensor_data_corrected
        else:
            tcp_in_base_T = self.get_eef_pose_in_base_frame()
            
            fts_force = np.matmul(tcp_in_base_T.matrix[:3, :3], sensor_data_corrected[:3])
            fts_torque = np.matmul(tcp_in_base_T.matrix[:3, :3], sensor_data_corrected[3:])
            
            return np.concatenate([fts_force, fts_torque])

