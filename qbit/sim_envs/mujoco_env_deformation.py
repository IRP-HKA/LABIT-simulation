"""
This Task Environment is used for both 
  - direct control of the robot for development 
  - batch execution for testing
"""

import time
import copy
import numpy as np
import grpc
from qbit.interfaces.grpc import qbit_pb2
from qbit.interfaces.grpc import qbit_pb2_grpc

import mujoco
import mujoco.viewer

from qbit.robots.ur5e_mj import UR5eMjArm
from qbit.robots.kuka_iiwa14_mj import KUKAiiwa14MjArm

from qbit.objects.object_base import DecomposedObject, MeshObject

from qbit.utils.tf_utils import T
from qbit.objects.env_objects import MjEnvObjects
from qbit.sim_envs.mujoco_env_base import MujocoEnvBase

from qbit.utils.mj_viewer_utils import update_view_camera_parameter

from qbit.interfaces.grpc.mj_grpc_proxy import QbitMjGrpcProxy


AVAILABLE_COMPONENTS = {
    'ur5e': UR5eMjArm,
    'iiwa14': KUKAiiwa14MjArm,
}



class MjEnvDeformation(MujocoEnvBase):

    
    def __init__(self,
                 task_env_config_path: str,
                 sim_timestep: float = 0.00001,
                 rendering_timestep: float = 0.033,
                 rt_factor: float = 0.0,
                 headless: bool = False,
                 server_modus: bool = True,
                 ):
        """
        In client_modus, the simulation will be run in server mode and can 
        be controlled by the client through the interface.
        """
        
        self._sim_timestep = sim_timestep
        self._rendering_timestep = rendering_timestep
        self._rt_factor = rt_factor
        self._headless = headless
        self._sim_time = 0
        self._last_render_time = 0
        
        _object_hole, _object_peg = self.load_env(task_env_config_path)
        
        self.z_insertion_depth = 0.0 #.001e-3 #(_object_hole.insertion_depth)/2
        self.insertion_goal_T = T(
            translation = [0.6, 0.0, _object_peg.insertion_depth/2 + _object_hole.insertion_depth - self.z_insertion_depth + 0.11],
            # translation = [0.6, 0.0, 0.0],
            quaternion = [0.707, 0.707, 0.0, 0.0]
        )
        
        
        self.compile_model()
        
        self.reset_sim()
        
        self._server_modus = server_modus
        if server_modus:
            self._interface = QbitMjGrpcProxy()
            self._interface.start()
            
        else:
            self.setup_client()

    
    def setup_client(self):
        
        # create the grpc client
        if self.check_is_server_aviailable('localhost:50052'):
            print("Creating client")
            self.channel = grpc.insecure_channel('localhost:50052')
            self.stub = qbit_pb2_grpc.QbitInterfaceStub(self.channel)
            response = self.stub.CheckServerConnection(qbit_pb2.Ping(ping=True))
        else:
            print("ERROR: Server not available")
            exit(0)


    def check_is_server_aviailable(self, target, timeout=2):
        """
        Check wether the insertion task service server is available
        """        
        channel = grpc.insecure_channel(target)
        try:
            # Wait until the channel is ready or timeout occurs.
            grpc.channel_ready_future(channel).result(timeout=timeout)
            return True
        except grpc.FutureTimeoutError:
            return False


    def __exit__(self):
        """
        Close the channel
        Make sure the grpc connection is closed properly and don't occupy the port
        """
        if not self._server_modus:
            if self.channel:
                self.stub.close()


    def get_random_start_and_goal_pose(self,
                                       pos_max_offset: float = 0.001, # meter
                                       rot_max_offset: float = 1.0, # degree
                                       zero_offset: bool = False,
                                       rotation_axis_offset: np.ndarray = np.array([0.0, 0.0, 0.15])
                                       ):
        """
        Get the random start and goal pose with the given offset range
        """
        if not zero_offset:
            pos_random_offset = np.random.uniform(-pos_max_offset, pos_max_offset, 2)
            rot_random_offset = np.random.uniform(-rot_max_offset, rot_max_offset, 3) * np.pi/180
        else:
            pos_random_offset = np.zeros(2)
            rot_random_offset = np.zeros(3)
        
        random_offset_T = T.from_euler(
            translation=[pos_random_offset[0], pos_random_offset[1], 0.0],
            euler=rot_random_offset.tolist()
        )
        rotation_axis_offset_T = T(
            translation=rotation_axis_offset,
        )

        goal_pose = self.insertion_goal_T * rotation_axis_offset_T * random_offset_T * rotation_axis_offset_T.inverse()

        start_pose = copy.deepcopy(goal_pose)

        start_pose.translation[2] += self.z_insertion_depth
        
        print(f"\n Randomized POS: {pos_random_offset} \n"
              f"Randomized ROT: {rot_random_offset * 180 / np.pi}")
        return start_pose, goal_pose
    
    
    def get_fixed_start_and_goal_pose(self,
                                      pos_offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
                                      rot_offset: np.ndarray = np.array([0.0, 0.0, 0.0]) * np.pi/180,
                                      rotation_axis_offset: np.ndarray = np.array([0.0, 0.0, 0.0])): #0.15
        """
        Get the fixed start and goal pose with the given offset
        """
        offset_T = T.from_euler(
            translation = pos_offset,
            euler = rot_offset,
        )
        rotation_axis_offset_T = T(
            translation=rotation_axis_offset,
        )
        goal_pose = self.insertion_goal_T * rotation_axis_offset_T * offset_T * rotation_axis_offset_T.inverse()
        start_pose = copy.deepcopy(goal_pose)

        start_pose.translation[2] += self.z_insertion_depth + 1.5e-3
        
        return start_pose, goal_pose


    def termination(self, 
                    current_eef_pose_T: T,
                    threshold: float = 0.0001
                    ) -> bool:

        if current_eef_pose_T.translation[2] < self.insertion_goal_T.translation[2] + threshold:
            return True
        if self.i >= 300:
            return True
        return False


    def spin_with_viewer(self):
        """
        Spin the simulation
        """
        with mujoco.viewer.launch_passive(self._mj_model, self._mj_data) as viewer:
            
            self.update_view_scale()
            update_view_camera_parameter(viewer)
            # self.update_view_opt(viewer)
            viewer.sync()
            
            while True:
                
                start_t = time.time()
                
                # check the received buffer
                if self._server_modus:
                    self.rec_action()

                self.robot.spin()
                
                self.step_mj_simulation()
                self._sim_time += self._sim_timestep
                
                self.update_interface_state_buffer()
                
                self.rendering(viewer)
                
                end_t = time.time()

                # sleep, if the simulation is faster than the desired real-time factor
                if self._rt_factor > 0:
                    sleep_time = self._sim_timestep * (1 / self._rt_factor) - (end_t - start_t)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        # print(f"Sleeping for {sleep_time * 1000} ms "
                        #       f"Max RT-Factor: {(sleep_time/(end_t-start_t)) + 1}")
                    else:
                        pass
                        # print(f"Execution exceeds desired period. "
                        #       f"Loop time: {(end_t - start_t) * 1000 } ms")



if __name__ == "__main__":
    
    task_env_config_path = "qbit/configs/envs/ur5e_deformation_task.yaml"
    
    mj = MjEnvDeformation(
        task_env_config_path=task_env_config_path,
        server_modus=True
        )
    mj.spin_with_viewer()
