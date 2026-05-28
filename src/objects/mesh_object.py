import mujoco
import numpy as np
from src.objects.base_object import BaseObject




class MeshObject(BaseObject):
    
    def __init__(self,
                 mj_spec,
                 config_dict: dict):
        super(MeshObject, self).__init__(mj_spec, config_dict)
        self.load_mesh_object(self._config)
        
    def load_mesh_object(self, config):
        
        if config.get('obj_name') in ["plate_benchmark", "housing_bottom", "housing_top", "housing_middle", "rotor_printed_part", "pcb"]:
            center_in_parent = np.array([0.0, 0.0, 0.0])
        else:
            center_in_parent = self.mesh_center

        geom_pos = -center_in_parent * 1

        # load the mesh files
        geom = self.obj_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_MESH,
            meshname = f"{config.get('obj_name')}_mesh",
            pos = geom_pos,
            condim = config.get('contact').get('condim', 3),
            rgba = config.get('mesh_color', [1, 0, 0, 1]),
            mass = self._obj_mass, #config.get('mass'),
            solref = self._materials[config.get('material')].solref,
            friction = self._materials[config.get('material')].friction,
        )

        mesh = self._mj_spec.add_mesh()
        mesh.name = f"{config.get('obj_name')}_mesh"
        mesh.file = config.get('mesh_path')
        mesh.scale = config.get('scale')

        print(f"loaded object {config.get('obj_name')}")
