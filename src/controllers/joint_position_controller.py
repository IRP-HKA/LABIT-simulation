"""
Arm position controller 
 - joint position control
 - eef position control
"""

import numpy as np


class JointPositionController:
    """
    PD joint position controller
    """

    def __init__(self,
                 kp: float = 1.0, # 30 for insertionnet, 1 for usb, 15 for admittance controller
                 kd: float = 0.02,
                 ki: float = 0.01,
                 control_loop_dt: float = 0.001,
                 joint_vel_max: np.ndarray = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]),
                 ):

        self._kp = kp
        self._kd = kd
        self._ki = ki
        self._control_loop_dt = control_loop_dt
        self._joint_vel_max = joint_vel_max * control_loop_dt
        
        self.sum_q_err = np.zeros(6)

    def pd_joint_position_control(self, q_start, q_goal, q_vel):
        """
        Copied from robot_base.py
        TODO:
        """
        
        q_pos_err = np.array(q_goal - q_start)
        
        # I-term
        self.sum_q_err += q_pos_err
        q_corr =  self._kp * q_pos_err + self._ki * self.sum_q_err + self._kd * q_vel
                
        # max_vel = self._joint_vel_max * self._control_loop_dt
        # q_corr = np.clip(q_corr, -max_vel, max_vel)

        q_cmd = q_corr
        
        # logger.warning(f"Q_START: {q_start}")
        # print(f"Q_POS_ERR: {q_pos_err}")
        # logger.warning(f"Q_CORRECTION: {q_corr}")
        # logger.warning(f"Q_CORRECTION CLIP: {q_corr}")
        # logger.warning(f"Q_KP: {self.KP * q_pos_err}")
        # logger.warning(f"Q_KD: {self.KD * q_vel}")      
        # logger.warning(f"Q_CMD: {q_cmd}")
        
        return q_cmd
