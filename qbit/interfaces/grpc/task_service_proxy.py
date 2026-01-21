
import time
import numpy as np
import copy
import grpc
from grpc import server
from concurrent import futures 
import threading

from qbit.interfaces.grpc import qbit_pb2
from qbit.interfaces.grpc import qbit_pb2_grpc

import keras


model_path = "/workspace/insertion/insertionnet/checkpoints/insertionnet_IF_sim_peg"


class _QbitTaskServicer(qbit_pb2_grpc.QbitInterfaceServicer):
    """
    This class serves as a gRPC servicer for the KubeBullet interface, providing 
    functionality for simulation and robot control in a Bullet physics environment.
    
    It implements the gRPC protobuf definitions, to handle incoming gRPC requests.

    """
    def __init__(self, task_cb=None) -> None:
        """
        :param cb (callable, optional): An optional callback function for extra processing.
        """
        
        # self.insertion_net = keras.saving.load_model(model_path, custom_objects=None, compile=True, safe_mode=True)

        self._task_cb = task_cb
        super().__init__()


    def CheckServerConnection(self, request, context):
        """
        Check connection status (TCP) for client before sending functional calls
        """
        print("Connection established")
        return qbit_pb2.Pong(pong = True)

    
    def RequestNextActionForInsertion(self, request, context):
        """
            Handles the gRPC request to determine the next action based on the received image and wrench data.
            Args:
                request (grpc.Request): The gRPC request containing image data and wrench data.
                context (grpc.Context): The gRPC context.
            Returns:
                qbit_pb2.EEFAction: The response containing the calculated end-effector action.
        """
        # print("Received image and wrench")
        
        start = time.time()
        img_bytes = request.image_data
        wrench = np.array(request.wrench)
        
        image_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape(256, 256, 3)
        img = np.expand_dims(image_array, 0)
        wrench = np.expand_dims(wrench, 0)
       
        # action = self.insertion_net([img, wrench])
        action = self._task_cb([img, wrench])
        action = action.numpy().flatten().tolist()

        print("Inference time [Server]: ", time.time()-start)

        return qbit_pb2.EEFAction(
            eef_action = action
        )


class QbitTaskService:
    """
    A gRPC server specifically designed to work with the Mujoco simulator.
    """
    
    def __init__(self,
                 insecure_port=50052,
                 max_workers=4) -> None:

        # create server with a thread pool executor
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

        # initialize the servicer
        self.servicer = _QbitTaskServicer(task_cb=self.inference_cb())
        
        # add servicer and add listen port
        qbit_pb2_grpc.add_QbitInterfaceServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f"[::]:{insecure_port}")

    def inference_cb(self):
        """
        Load the model and return the callback function for inference.
        """
        insertion_net = keras.saving.load_model(
            model_path,
            custom_objects=None,
            compile=True,
            safe_mode=True)
        return insertion_net        

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
    server = QbitTaskService()
    server.start()
    
    print(threading.current_thread())
