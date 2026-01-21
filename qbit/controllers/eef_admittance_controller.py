"""
EEF Force Controller
"""

import numpy as np

from qbit.utils.tf_utils import T, rodrigues_rotation


class EEFAdmittanceController:
    """
    End-effector admittance force controller
     - Calculate the next EEF pose based on the current wrench
    """

    def __init__(self,
                vel_max_lin: float = 0.1,
                vel_max_rot: float = 10,
                control_loop_dt = 0.002,
                eef_lin_gain = 10,
                eef_rot_gain = 10,
                target_wrench = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0]),
                M = np.array([5.0, 5.0, 5.0, 3, 3, 3]) / 5, # * 10
                C = np.array([1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0]),
                D = np.array([400.0, 400.0, 400.0, 800.0, 800.0, 800.0]) / 5,
                activation_mask = np.array([1, 1, 1, 1, 1, 1]),
                ):
        
        self.control_loop_dt = control_loop_dt
        self._vel_max_lin = vel_max_lin   # m/s, higher value gripper approach too fast if current force=0
        self._vel_max_rot = vel_max_rot  # rad/s    
        self.eef_lin_gain = eef_lin_gain
        self.eef_rot_gain = eef_rot_gain
        
        # admittance control
        # self.activation_mask = np.array([1, 0, 0, 0, 0, 0])
        self.activation_mask = activation_mask
        
        self.M = M # Parameter for push in z direction
        # self.M = np.array([5.0, 5.0, 25.0, 3, 3, 3])  # Parameter for push in z direction 
        self.C = C
        self.D = D # np.array([400.0, 400.0, 400.0, 800.0, 800.0, 800.0])  / 5
        # self.D = np.array([100.0, 100.0, 400.0, 7.0, 7.0, 7.0]) # * 80

        # initial twist
        self.xdd = np.zeros(6)
        self.calculated_twist = np.zeros(6) # calculated twist from admittance control

        # initial wrench
        self.target_wrench = target_wrench
        self.actual_wrench = np.zeros(6)


    def admittance_control(self, actual_wrench,
                           current_eef_pose: T,
                           ):

        self.actual_wrench = np.array(actual_wrench)
        self.error_wrench = (self.target_wrench + self.actual_wrench)
        
        # calculate acceleration
        self.xdd = (1/self.M) * (self.error_wrench - self.D * self.calculated_twist)

        # print("current twist: ", self.calculated_twist[:6])
        # print("xdd * dt: ", self.xdd[:6] * self.control_loop_dt)
        
        self.calculated_twist = self.calculated_twist + self.xdd * self.control_loop_dt

        self.calculated_twist = self.clip_twist(self.calculated_twist)
        
        # activate / deactivate axes
        self.target_twist = self.select_axes(self.calculated_twist)
        self.target_twist = self.clip_twist(self.target_twist)

        v_tran = self.target_twist[:3]
        w_rot = self.target_twist[3:]
        
        next_position = current_eef_pose.translation + v_tran * self.control_loop_dt  * self.eef_lin_gain
        
        # Next orientation
        delta_rot = rodrigues_rotation(w_rot, self.control_loop_dt * self.eef_rot_gain)
        next_orientation = current_eef_pose.matrix[:3, :3] @ delta_rot
        # next_orientation = current_eef_pose.euler_rotation + w_rot * self.control_loop_dt * 10
        
        eef_goal_T = T.from_tran_and_rotaion_matrix(next_position, next_orientation)
        
        # print("NEW TARGET TWIST: ", self.target_twist[:6])
        # print("NEW ORIENTATION: ", next_orientation)
        # print("Next position: ", next_position)
        # print("Next orientation: ", next_orientation)

        return eef_goal_T

        
    def select_axes(self, twist):
        """
        Select axes to control
        """
        return self.activation_mask * twist


    def clip_twist(self, twist):
        """
        Clip the twist to the maximum velocity
        """
        twist[0:3] = np.clip(twist[0:3], -self._vel_max_lin, self._vel_max_lin)
        twist[3:6] = np.clip(twist[3:6], -self._vel_max_rot, self._vel_max_rot)
        return twist
