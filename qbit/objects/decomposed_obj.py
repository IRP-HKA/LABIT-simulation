"""
Decomposed object class for adding decomposed object (meshes) to the MuJoCo model
"""

import os
import glob
import numpy as np

import mujoco
import mujoco.viewer

from qbit.objects.object_base import ObjectBase


class DecomposedObj(ObjectBase):
    """
    Add the decomposed object (meshes) to the MuJoCo model
    """

    def __init__(self, mj_spec, obj_body_name, mesh_path, mesh_type):
        super().__init__(mj_spec, obj_body_name, mesh_path, mesh_type)
        
        self.load_mesh()
    
    def load_mesh(self):
        self.mesh_files = sorted(glob.glob(os.path.join(self.mesh_path, "*.obj")))
    
    def attach_meshes_to_body(self,
                              parent_body, #str,
                              pos = [0, 0, 0],
                              qua = [1, 0, 0, 0],
                              mesh_color = [0, 0, 1, 1],
                              mesh_name_prefix = "female",
                              mass = 0.0001):
        # mesh_color = np.random.rand(3).tolist() + [1.0]  # Random RGB color with alpha = 1.0
        obj_body = parent_body.add_body(
            name = self._obj_body_name,
            pos = pos,
            quat = qua,    
            mass = mass,
        )

        for i, f in enumerate(self.mesh_files):
            mesh_color = np.random.rand(3).tolist() + [1.0]  # Random RGB color with alpha = 1.0
            geom = obj_body.add_geom(
                type = mujoco.mjtGeom.mjGEOM_MESH,
                meshname = f'{mesh_name_prefix}_{i}',
                # quat = [0.7071068, 0.7071068, 0, 0],
                # quat = qua,
                rgba = mesh_color
            )
            mesh = self._mj_spec.add_mesh()
            mesh.name = f'{mesh_name_prefix}_{i}'
            mesh.file = f
            mesh.scale = np.array([0.001, 0.001, 0.001]) * 3
