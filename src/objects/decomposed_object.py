import glob
import os
import mujoco

from src.objects.base_object import BaseObject



class DecomposedObject(BaseObject):
    
    def __init__(self,
                 mj_spec,
                 config_dict: dict,):
        super(DecomposedObject, self).__init__(mj_spec, config_dict)

        _mp = self._materials(obj_path=self._config.get('mesh_path'))
        if self._config.get('mesh_type') == 'vhacd':
            _mp.decomposition_with_vhacd()
        elif self._config.get('mesh_type') == 'coacd':
            _mp.decomposition_with_coacd(threshold=0.01)
        self._decomposed_mesh_dir = _mp._decomposed_mesh_dir

        self.load_decomposed_object(config=self._config)

    def load_decomposed_object(self, config):
        
        # load the mesh files
        mesh_files = sorted(glob.glob(os.path.join(self._decomposed_mesh_dir, "*.obj")))
        mesh_color = config.get('mesh_color', [1, 0, 0, 1]),
        for i, f in enumerate(mesh_files):
            # mesh_color = [0, 0, 1, 1]
            # mesh_color = np.random.rand(3).tolist() + [1.0]  # Random RGB color with alpha = 1.0
            geom = self.obj_body.add_geom(
                type = mujoco.mjtGeom.mjGEOM_MESH,
                meshname = f"{config.get('obj_name')}_mesh_{i}",
                condim = config.get('contact').get('condim', 3),
                rgba = mesh_color[0],
                density = self._materials[config.get('material')].density,
                solref = self._materials[config.get('material')].solref,
                friction = self._materials[config.get('material')].friction,
            )
            mesh = self._mj_spec.add_mesh()
            mesh.name = f"{config.get('obj_name')}_mesh_{i}"
            mesh.file = f
            mesh.scale = config.get('scale')

        print("loaded mesh files")


