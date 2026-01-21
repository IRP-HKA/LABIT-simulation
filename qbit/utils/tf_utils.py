import numpy as np
from qbit.utils import tf_transformation


class T:
    """
    Transformation matrix class for coordinate transformation
    """
    def __init__(self,
                 translation=(0.0, 0.0, 0.0),
                 quaternion=(0.0, 0.0, 0.0, 1.0),
                 frame_name=None,
                 parent_frame=None,
                 name=None,
                 ) -> None:
        
        self.frame_name = frame_name
        self.parent_frame = parent_frame
        self.name = name
        
        self._matrix = np.zeros((4, 4))
        self._matrix[3, 3] = 1
        
        self._matrix[:3, 3] = np.array(translation)
        self._matrix[:3, :3] = np.array(
            tf_transformation.quaternion_matrix(quaternion)[:3, :3]
        )

    @classmethod
    def from_euler(cls,
                   translation,
                   euler: list,
                   axes='sxyz'):
        
        quaternion = tf_transformation.quaternion_from_euler(*euler, axes=axes)
        
        return cls(translation, quaternion)
    
    @classmethod
    def from_matrix(cls, matrix):
        t = cls()
        t._matrix = matrix
        return t

    @classmethod
    def from_tran_and_rotaion_matrix(cls, translation, rotation_matrix):
        t = cls()
        t._matrix[:3, 3] = translation
        t._matrix[:3, :3] = rotation_matrix
        return t
    
    @property
    def translation(self) -> np.ndarray:
        """
        get the translation
        """
        return self._matrix[:3, 3]

    @property
    def matrix(self) -> np.ndarray:
        """
        get the transformation matrix
        """
        return self._matrix
    
    @matrix.setter
    def matrix(self, matrix) -> None:
        self._matrix = matrix
        return
    
    @translation.setter
    def translation(self, translation):
        self._matrix[:3, 3] = translation
        return
    
    def inverse(self):
        # translation = -self.matrix[:3, 3]
        return T.from_matrix(np.linalg.inv(self._matrix))
    
    @property
    def quaternion(self):
        """
        get the quaternion
        """
        qua = tf_transformation.quaternion_from_matrix(self._matrix[:4, :4])
        
        return qua
    
    @property
    def euler_rotation(self) -> np.ndarray:
        """
        get the rotation in euler angles
        """
        qua = tf_transformation.quaternion_from_matrix(self._matrix[:4, :4])
        euler = tf_transformation.euler_from_quaternion(qua)

        return np.array(euler)
    
    def get_pos_quat_list(self, quat_format='xyzw') -> list:
        """
        Get the position and quaternion as a list
        """
        if quat_format == 'xyzw':
            return self.translation.tolist() + self.quaternion.tolist()
        elif quat_format == 'wxyz':
            return self.translation.tolist() + [self.quaternion[3]] + self.quaternion[:3].tolist()
        # return self.translation.tolist() + self.quaternion.tolist()
    
    def __repr__(self) -> str:
        return f"Translation: {self.translation} || Quaternion: {self.quaternion}"
      
    def __str__(self) -> str:
        return f"Translation: {self.translation} || Quaternion: {self.quaternion}"

    def __mul__(self, second_T):
        return T.from_matrix(self.matrix @ second_T.matrix)


def orientation_error(R_des, R_curr):
    """
    Compute a 3-element orientation error between a desired rotation matrix and the current one.
    Uses the vee operator on the skew-symmetric part of R_des @ R_curr^T.
    """
    R_err = R_des @ R_curr.T
    # The error vector is half the difference between the off-diagonal elements.
    err = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])
    return err


def hat(w):
    """
    Compute the skew-symmetric matrix of a 3D vector.
    """
    return np.array([[    0, -w[2],  w[1]],
                     [ w[2],     0, -w[0]],
                     [-w[1],  w[0],     0]])
    
def rodrigues_rotation(w, dt):
    """
    Compute the rotation matrix corresponding to the exponential map
    of the skew-symmetric matrix of angular velocity w scaled by dt,
    using Rodrigues' formula.
    
    Parameters:
      w: 3-element angular velocity vector.
      dt: time step.
    
    Returns:
      A 3x3 rotation matrix.
    """
    theta = np.linalg.norm(w) * dt
    if theta < 1e-6:
        # For very small angles, return an approximation.
        return np.eye(3) + hat(w) * dt
    else:
        k = w / np.linalg.norm(w)  # unit vector
        K = hat(k)
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
