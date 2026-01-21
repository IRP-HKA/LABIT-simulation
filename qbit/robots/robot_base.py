
from typing import Tuple
import numpy as np

import mujoco
from mujoco import mjtObj
import mujoco.viewer

from qbit.utils.tf_utils import T
from qbit.utils.mujoco_utils import print_object_names
from qbit.controllers.joint_position_controller import JointPositionController
from qbit.controllers.eef_position_controller import EEFPositionController


class RobotBase:
    """
    Because MuJoCo is a physics engine and doesn't support dynamically importing separate models,
    like in pybullet during runtime, we need to manually manage the robot models and all assets.
    
    Before running the simulation, we need to merge all the assets into a single XML file.
    
     - find controllable joint indexes
     - initialize: hard reset joint position and set joint position controller
     - execute_motion_primitive: primitive need to be implemented in child class
     - joint_position_controller
     - joint_velocity_controller
    """

    def __init__(self,
                 qbit_robot_config: dict,
                 headless: bool = False,
                 arm_base_pos = (0.2, 0, 0.95),
                 arm_base_qua = (0, 0, 0, -1), # wxyz # TODO: Use the default base quaternion in xml, check and change this later
                 ) -> None:

        self._config = qbit_robot_config
        self._headless = headless
        self._sim_time = 0

        self._arm_base_pos = arm_base_pos
        self._arm_base_qua = arm_base_qua
        
        self._mj_spec = mujoco.MjSpec()
        
        self._mj_spec.from_file(self._config.get('mujoco_xml_path'))

        self.reset_base_pose(arm_base_pos, arm_base_qua)
        
        # attributes
        self._status = 'loaded'

        self.load_controllers()
        # self.check_xml()
        
        self._goal_joint_pos = np.zeros(6)
        
        
    def load_controllers(self):
        
        for key, controller in self._config.get('controllers', {}).items():

            if key == 'joint_position_controller':
                self._joint_position_controller = JointPositionController(
                    kp = controller.get('kp'),
                    kd = controller.get('kd'),
                    ki = controller.get('ki'),
                    control_loop_dt = controller.get('control_loop'),
                    joint_vel_max = np.array(controller.get('vel_max'))
                )
                print("Joint position controller loaded")
            elif key == 'eef_position_controller':
                self._eef_position_controller = EEFPositionController(
                    kp = controller.get('kp'),
                    kd = controller.get('kd'),
                    control_loop_dt = controller.get('control_loop'),
                    eef_pos_vel_max = np.array(controller.get('vel_max'))
                )
                pass
            
    
    def get_current_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current joint state
        """
        # first six joint states correspond to robot joints
        return (
            self._mj_data.qpos[:6],
            self._mj_data.qvel[:6]
        )

    
    def update_goal_joint_pos(self, goal_joint_pos: np.ndarray):
        """
        Update the goal joint position
        """
        self._goal_joint_pos = goal_joint_pos
        return
    
    def spin(self):
        """
        Run low-level position control loop
        """
        return NotImplementedError

    def reset(self):
        """
        Move the robot to the desired start pose: for example same as the UR5e in the lab.
        Advance simulation in two steps: before external force and control is set by user.
        """
        # self._goal_joint_pos = np.array([0, -1.57, 1.57, -1.57, -1.57, 60.0/180 * np.pi]) #-1.1854278979642656
        self._goal_joint_pos = np.array([-0.22403618026677777, -2.0, 1.7770384166108946, 
                                         -2.1624069751818045, -1.57079592802717, 2.9175564735513344])
        self._mj_data.qpos[0: 6] = self._goal_joint_pos
        self._mj_data.ctrl[0: 6] = self._goal_joint_pos
        
        mujoco.mj_step1(self._mj_model, self._mj_data)
        return
    
    def update_mj_pointer(self, mj_model, mj_data):
        """
        Update the mj_data and mj_model pointer
        """
        self._mj_model = mj_model
        self._mj_data = mj_data
        return
    
    def reset_base_pose(self,
                        base_pos: Tuple[float, float, float],
                        base_qua: Tuple[float, float, float, float]) -> None:
        """
        Reset the base pose of the robot
        """
        self._mj_spec.find_body('base').pos = base_pos
        self._mj_spec.find_body('base').quat = base_qua
        return
    
    @property
    def mj_spec(self):
        """
        Return the MuJoCo model specification for adding the environment and task objects.
        """
        return self._mj_spec


    def check_xml(self):
        """
        Check the xml file for the robot
        """
        print_object_names(self._mj_model,
                           mjtObj.mjOBJ_BODY, self._mj_model.nbody, "Bodies")
        print_object_names(self._mj_model,
                           mjtObj.mjOBJ_JOINT, self._mj_model.njnt, "Joints")
        # print_object_names(self._mj_model,
        #                  mjtObj.mjOBJ_GEOM, self._mj_model.ngeom, "Geoms")
        print_object_names(self._mj_model,
                           mjtObj.mjOBJ_SITE, self._mj_model.nsite, "Sites")
        print_object_names(self._mj_model,
                           mjtObj.mjOBJ_SENSOR, self._mj_model.nsensor, "Sensors")
        print_object_names(self._mj_model,
                           mjtObj.mjOBJ_ACTUATOR, self._mj_model.nuser_actuator, "User defined actuators")
        
        print(f"Sensor numbers: {self._mj_model.nsensordata} ")
        return
    
    
    def move_to_joint_position(self):
        """
        Be implemented in the child class depending on the robot type
        """
        return NotImplementedError
    
    
    def move_to_eef_position(self):
        """
        Be implemented in the child class depending on the robot type
        """
        return NotImplementedError
    