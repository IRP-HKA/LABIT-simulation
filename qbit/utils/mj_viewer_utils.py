
import numpy as np
import mujoco

def update_view_camera_parameter(viewer, view_type = "default"):
    """
    Updates the camera parameters of the viewer based on the specified view type.
    Parameters:
        viewer (mujoco.Viewer): The viewer object whose camera parameters are to be updated.
        view_type (str, optional): The type of view to set the camera parameters for. 
            Default is "default". Other options include "mesh_comparision" and "mesh_usb".
    """

    # Default view
    viewer.cam.lookat[:] = np.array([-0.51057842, -0.39353146,  0.99606974])
    viewer.cam.distance = 0.8367947840836062
    viewer.cam.azimuth = -99.25
    viewer.cam.elevation = -7.4375
    
    # View for mesh comparison
    if view_type == "mesh_comparision":

        viewer.cam.lookat[:] = np.array([-0.63762056, -0.23479806, -0.64429904])
        viewer.cam.distance = 1.92430391572589
        viewer.cam.azimuth = -134.40795781399808
        viewer.cam.elevation = -79.13153163950155

    if view_type == "mesh_usb":
        viewer.cam.lookat[:] = np.array([-0.61754595, -0.32904063,  0.71167])
        viewer.cam.distance = 0.716194258999645
        viewer.cam.azimuth = -125.125
        viewer.cam.elevation = -38.75
    
    if view_type == "labit_benchmark":
        viewer.cam.lookat[:] = np.array([-0.317,0.110, 1.032])
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = -45.0
        viewer.cam.elevation = -45.0
        
    return

