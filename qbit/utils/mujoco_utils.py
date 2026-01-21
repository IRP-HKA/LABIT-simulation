from mujoco import mj_id2name, mj_name2id
from mujoco import mjtObj
import numpy as np
from qbit.utils.tf_utils import T


def print_object_names(model, obj_type, count, label):
    """
    Prints the names for objects of a given type from the model.
    """
    print(f"{label}:")
    for i in range(count):
        # Retrieve the name of the object using its type and id.
        name = mj_id2name(model, obj_type, i)
        if name is not None:
            print(f"  - {name}")
        else:
            print(f"  - [unnamed] (id {i})")


def convert_quat_to_wxyz(quat):
    """
    Convert the quaternion format between wxyz for mujoco and xyzw.
    """
    return [quat[3], quat[0], quat[1], quat[2]]
    
def convert_quat_to_xyzw(quat):
    """
    Convert the quaternion format between wxyz for mujoco and xyzw.
    """
    return [quat[1], quat[2], quat[3], quat[0]]

def get_body_pose_in_world(mj_model, mj_data, body_name):
    """
    Get the body pose in the world frame.
    """
    body_id = mj_name2id(mj_model, mjtObj.mjOBJ_BODY.value, body_name)

    pos = mj_data.xpos[body_id, :]
    quat = mj_data.xquat[body_id, :]  # wxyz

    return pos, quat

def get_relative_pose(mj_model, mj_data, body_name_a, body_name_b, ensure_negative_z_axis = True):
    """
    Get the pose of object b in frame of object a.
    """
    pos_a, quat_a = get_body_pose_in_world(mj_model, mj_data, body_name_a)
    pos_b, quat_b = get_body_pose_in_world(mj_model, mj_data, body_name_b)

    pose_a_world = T(translation=pos_a, quaternion=convert_quat_to_xyzw(quat_a))._matrix
    pose_b_world = T(translation=pos_b, quaternion=convert_quat_to_xyzw(quat_b))._matrix

    # make sure that z-axis of object is always showing downward.
    if pose_b_world[2, 2] > 0 and ensure_negative_z_axis:
        pose_b_world[:3,:3] = pose_b_world[:3,:3] @ np.diag([1.0, -1.0, -1.0])

    pose_b_a = np.linalg.inv(pose_a_world) @ pose_b_world

    return T.from_matrix(pose_b_a)
