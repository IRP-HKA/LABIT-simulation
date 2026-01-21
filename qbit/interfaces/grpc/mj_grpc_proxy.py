

import numpy as np
import copy
import grpc
from grpc import server
from concurrent import futures 
import threading

from qbit.interfaces.grpc import qbit_pb2
from qbit.interfaces.grpc import qbit_pb2_grpc



class _MjGrpcServicer(qbit_pb2_grpc.QbitInterfaceServicer):
    
    def __init__(self, cb=None) -> None:
        """
        :param cb (callable): callback function to the task service.
        """

        self._cb = cb
        
        self._cmd_buffer = {
            'cmd_type': None,
            'cmd': None
        }
        
        self._robot_state_buffer = {
            'joint_pos': [],
            'joint_vel': [],
            'eef_pose': []
        }
        
        self._rec_new_cmd = False
        
        super().__init__()
        
    
    def CheckServerConnection(self, request, context):
        """
        Check connection status (TCP) for client before sending functional calls
        """
        print("Received handshake, establish connection.")
        return qbit_pb2.Pong(pong = True)

    
    def GetArmJointState(self, request, context):
        """
        Get the current joint state of the robot
        """
        return qbit_pb2.ArmJointState(
            positions = self._robot_state_buffer['joint_pos'],
            velocities = self._robot_state_buffer['joint_vel']
        )
        
        
    def MoveArmToJointPos(self, request, context):
        """
        Send the joint positions to the robot
        Like a move command for the real robot
        """
        self._cmd_buffer = {
            'cmd_type': 'joint_pos',
            'cmd_data': np.array(request.joint_pos)
        }
        self._rec_new_cmd = True
        
        print("Received joint positions: ")
        print(self._cmd_buffer)
        
        return qbit_pb2.ArmJointState(
            positions = self._robot_state_buffer['joint_pos'],
            velocities = self._robot_state_buffer['joint_vel']
        )


    def MoveArmEEFtoPose(self, request, context):
        """
        Send the EEF Pose in robot base frame to the robot
        Like a move command for the real robot
        """
        eef_pose = self.decode_eef_pose_cmd_to_ndarray(request)
        
        self._cmd_buffer = {
            'cmd_type': 'eef_pose',
            'cmd_data': eef_pose
            }
        self._rec_new_cmd = True
        
        print("Received EEF pose: ")
        print(self._cmd_buffer)
        
        return qbit_pb2.ArmJointState(
            positions = self._robot_state_buffer['joint_pos'],
            velocities = self._robot_state_buffer['joint_vel']
        )

    @staticmethod
    def decode_eef_pose_cmd_to_ndarray(pose_cmd):
        """
        Convert the gRPC pose to a numpy array.
        """
        position = [
            pose_cmd.eef_position.x,
            pose_cmd.eef_position.y,
            pose_cmd.eef_position.z
        ]
        quaternion = [
            pose_cmd.eef_quaternion.x,
            pose_cmd.eef_quaternion.y,
            pose_cmd.eef_quaternion.z,
            pose_cmd.eef_quaternion.w
        ]
        return position + quaternion


class QbitMjGrpcProxy:
    """
    A gRPC server/client specifically designed to work with the Mujoco simulator.
    """
    
    def __init__(self,
                 insecure_port: int = 50052,
                 max_workers: int = 4,
                 server_modus: bool = True) -> None:
        
        if server_modus:
            # create server with a thread pool executor
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
            
            # initialize the servicer
            self.servicer = _MjGrpcServicer(cb=self.cb_start_simulation)
            
            # add servicer and add listen port
            qbit_pb2_grpc.add_QbitInterfaceServicer_to_server(self.servicer, self.server)
            self.server.add_insecure_port(f"[::]:{insecure_port}")


    def check_rec_cmd_buffer(self):
        """
        check the command buffer
        """
        if self.servicer._rec_new_cmd:
            self.servicer._rec_new_cmd = False        
            return [True, self.servicer._cmd_buffer]
        else:
            return [False, None]

    def update_robot_state_buffer(self,
                                  joint_state,):

        self.servicer._robot_state_buffer = {
            'joint_pos': copy.deepcopy(joint_state[0]),
            'joint_vel': copy.deepcopy(joint_state[1]),
        }
        return


    def start_server(self):
        """Start the server"""
        self.server.start()
        self.server.wait_for_termination()


    def start(self):
        """Start thread"""
        thread = threading.Thread(target=self.start_server)
        thread.start()
        print("Started thread for gRPC server")

    
    def cb_start_simulation(self):
        """Call back -> TODO"""
        print("Receiving Request to start the simulation")
        return


if __name__ == "__main__":
    server = QbitMjGrpcProxy()
    server.start()
    print(threading.current_thread())
