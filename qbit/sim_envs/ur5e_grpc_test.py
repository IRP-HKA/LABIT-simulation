"""
Test script for the qbit interface using grpc to control the robot in MuJoCo.
"""

import time
import grpc

from qbit.interfaces.grpc import qbit_pb2
from qbit.interfaces.grpc import qbit_pb2_grpc


def run():

    try:
        with grpc.insecure_channel('localhost:50052') as channel:
            stub = qbit_pb2_grpc.QbitInterfaceStub(channel)
            
            response = stub.CheckServerConnection(qbit_pb2.Ping(ping=True))
            print("Client received: ", response.pong)

            start = time.time()

            # joint position
            # response = stub.MoveArmToJointPos(qbit_pb2.ArmJointPosCmd(
            #     joint_pos=np.array([0.3, -1.07, 1.67, -1.57, -1.57, 60.0/180 * np.pi])
            #     ))
            # print(response.positions)
            # print(response.velocities)
            
            # eef pose
            # print("EEF Pose")
            # eef_pose_cmd = qbit_pb2.ArmEEFPoseCmd(
            #     eef_position=qbit_pb2.position(x=0.6, y=0.0, z=0.238),
            #     eef_quaternion=qbit_pb2.quaternion(x=0.707, y=0.707, z=0.0, w=0.0)
            # )
            # response = stub.MoveArmEEFtoPose(eef_pose_cmd)
            # print(response.positions)
            
            response = stub.GetArmJointState(qbit_pb2.Ping(ping=True))
            print(response.positions)
            print(response.velocities)

            print("Inference time [Client]: ", time.time()-start)
            
    except Exception as e:
        print("An error occurred:", e)
    
if __name__ == '__main__':
    run()
