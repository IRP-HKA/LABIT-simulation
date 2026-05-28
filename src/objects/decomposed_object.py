import glob
import os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as _R

from src.objects.base_object import BaseObject
from src.utils.mesh_processing import MeshObjects



class DecomposedObject(BaseObject):

    def __init__(self,
                 mj_spec,
                 config_dict: dict,):
        super(DecomposedObject, self).__init__(mj_spec, config_dict)

        _mp = MeshObjects(obj_path=self._config.get('mesh_path'))
        if self._config.get('mesh_type') == 'vhacd':
            _mp.decomposition_with_vhacd()
        elif self._config.get('mesh_type') == 'coacd':
            _mp.decomposition_with_coacd(threshold=0.01)
        self._decomposed_mesh_dir = _mp._decomposed_mesh_dir

        self.load_decomposed_object(config=self._config)

    def load_decomposed_object(self, config):

        # Compute the same centroid offset that attach_body added to body.pos,
        # so geoms are placed at attach_pose.position + vertex (matching MeshObject).
        quat = config.get("attach_pose", {}).get("quaternion", [1, 0, 0, 1])
        R_mat = _R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        if config.get('obj_name') in self.objects_to_not_center_cs:
            center_in_parent = np.array([0.0, 0.0, 0.0])
        else:
            center_in_parent = R_mat @ self.mesh_center

        # load the mesh files
        mesh_files = sorted(glob.glob(os.path.join(self._decomposed_mesh_dir, "*.obj")))
        mesh_color = config.get('mesh_color', [1, 0, 0, 1])
        for i, f in enumerate(mesh_files):
            geom = self.obj_body.add_geom(
                type = mujoco.mjtGeom.mjGEOM_MESH,
                meshname = f"{config.get('obj_name')}_mesh_{i}",
                pos = -center_in_parent,
                condim = config.get('contact').get('condim', 3),
                rgba = mesh_color,
                density = self._materials[config.get('material')].density,
                solref = self._materials[config.get('material')].solref,
                friction = self._materials[config.get('material')].friction,
            )
            mesh = self._mj_spec.add_mesh()
            mesh.name = f"{config.get('obj_name')}_mesh_{i}"
            mesh.file = f
            mesh.scale = config.get('scale')

        print("loaded mesh files")


