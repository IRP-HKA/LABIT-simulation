import mujoco
from src.objects.base_object import BaseObject

class SDFObject(BaseObject):
    
    def __init__(self,
                 mj_spec,
                 config_dict: dict):
        super(SDFObject, self).__init__(mj_spec, config_dict)
        self.load_mesh_object(self._config)
        
    def load_mesh_object(self, config):

        # load the mesh files
        geom = self.obj_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_SDF,
            meshname = f"{config.get('obj_name')}_mesh",
            condim = config.get('contact').get('condim', 3),
            rgba = config.get('mesh_color', [1, 0, 0, 1]),
            mass = self._obj_mass,
            solref = self._materials[config.get('material')].solref,
            friction = self._materials[config.get('material')].friction,
        )
        geom.plugin.instance_name = "sdf1"
        geom.plugin.active = 1
        
        mesh = self._mj_spec.add_mesh()
        mesh.name = f"{config.get('obj_name')}_mesh"
        mesh.file = config.get('mesh_path')
        mesh.scale = config.get('scale')
        mesh.plugin.instance_name = "sdf1"

        print(f"loaded object {config.get('obj_name')}")

