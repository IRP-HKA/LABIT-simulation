"""
Arm position controller 
 - joint position control
 - eef position control
"""

import copy
import numpy as np
from tracikpy import TracIKSolver
from qbit.utils.tf_utils import T


class EEFPositionController:
    """
    Interpolate the end-effector position and orientation
    """

    EEF_KP_POS = 5
    EEF_KD_POS = 2
    EEF_KP_ROT = 2
    EEF_KD_ROT = 0.2
    
    def __init__(self,
                 kp: float = 2, # 30 for insertionnet, 1 for usb, 15 for admittance controller
                 kd: float = 0.02,
                 control_loop_dt: float = 0.001,
                 eef_pos_vel_max: np.ndarray = np.array([1.0, 1.0, 1.0]),
                 ):

        self._kp = kp
        self._kd = kd
        self._control_loop_dt = control_loop_dt
        self._eef_pos_vel_max = eef_pos_vel_max
        
        self.sum_q_err = np.zeros(6)
        
        # TODO: parse using xml file
        self.tcp_id = 0
        self.base_link_name = 'base'
        self.base_id = 1
        self.tcp_body_name = 'tool0'
        self.tcp_id = 8
    
        self.ik = TracIKSolver(
                "/workspace/qbit/assets/robots/ur5e/ur5e_robot.urdf",
                # "/workspace/qbit/assets/robots/ur5e/ur5e_robot_calibrated.urdf",
                base_link="base_link",
                tip_link="flange",
                timeout=0.05
            )

    def eef_position_control(self,
                             current_eef_pose,
                             target_eef_pose,
                             q_init,
                             return_q_cmd=True):
        
        """
        EEF position control
        Return the joint command calculated by the considering the max eef velocity
         - Linear interpolation of the eef position
        """
        
        tcp_in_base_T = current_eef_pose
        
        tcp_position = tcp_in_base_T.translation
        tcp_pos_err = target_eef_pose.translation - tcp_position
        
        # allowed_step = self._eef_pos_vel_max * self._control_loop_dt
        # step = np.sign(tcp_pos_err) * np.minimum(np.abs(tcp_pos_err), allowed_step)
                
        tcp_pos_cmd = tcp_position + tcp_pos_err #step
        tcp_cmd_T = copy.deepcopy(target_eef_pose)
        tcp_cmd_T.translation = tcp_pos_cmd
        
        # print("Target eef euler: ", target_eef_pose.euler_rotation)
        # print("Current tcp position: ", tcp_position)
        # print("Step: ", step)
        # print("CMD eef position: ", tcp_cmd_T)
        # print("--------------------")

        if not return_q_cmd:
            return tcp_cmd_T
        else:
            q_cmd = self.ik.ik(tcp_cmd_T.matrix, q_init.tolist())
            return q_cmd


    def pd_eef_position_control(self, eef_current, eef_goal, eef_last, q_init):
        
        eef_pos_err = eef_goal.translation - eef_current.translation
        eef_rot_err = eef_goal.euler_rotation - eef_current.euler_rotation
        
        eef_pos_corr = self.EEF_KP_POS * eef_pos_err + self.EEF_KD_POS * (eef_current.translation - eef_last.translation)
        eef_rot_corr = self.EEF_KP_ROT * eef_rot_err + self.EEF_KD_ROT * (eef_current.euler_rotation - eef_last.euler_rotation)
        
        max_pos_vel = self.EEF_VEL_MAX * self._control_loop_dt
        max_rot_vel = self.EEF_VEL_MAX * self._control_loop_dt
        
        eef_pos_corr = np.clip(eef_pos_corr, -max_pos_vel, max_pos_vel)
        eef_rot_corr = np.clip(eef_rot_corr, -max_rot_vel, max_rot_vel)
        
        eef_pos_cmd = eef_current.translation + eef_pos_corr
        eef_rot_cmd = eef_current.euler_rotation #+ eef_rot_corr
        
        eef_cmd_T = T.from_euler(eef_pos_cmd, eef_rot_cmd)
        
        q_cmd = self.ik.ik(eef_cmd_T.matrix, q_init.tolist())
        
        # print("--------------------")
        # print("EEF_CURRENT: ", eef_current.translation)
        # print("EEF_POS_ERR: ", eef_pos_err)
        # print("EEF_POS_CORR: ", eef_pos_corr)
        # print("EEF_POS_CMD: ", eef_pos_cmd)

        return q_cmd
