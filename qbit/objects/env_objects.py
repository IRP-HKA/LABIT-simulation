

import mujoco


class MjEnvObjects:
    
    def __init__(self,
                 mj_spec,
                 env_objects: dict
                 ):
        self._mj_spec = mj_spec
        
        for env_obj in env_objects:
            self.load_env_objects(env_obj)


    def load_env_objects(self, 
                         env_obj: dict):
        """
        Add the environment objects as site object to the MuJoCo model
        Example yaml file:
            env_objects:
            - obj_name: table
                mesh_type: mesh # mesh / sphere / coacd
                mesh_path: /workspace/qbit/assets/task_env/lab_tables/tisch_neu_s0.obj
                attach_body: 'world'
                attach_pose:
                position: [0, -0.35, 0.15]
                quaternion: [ 0.5, 0.5, 0.5, 0.5 ]
                scale: [1, 1, 1]
        """
        
        obj_name = env_obj.get('obj_name')
        mesh_path = env_obj.get('mesh_path')
        
        site_body = self._mj_spec.worldbody.add_body(
            name = f"{obj_name}_body",
            pos = env_obj.get('attach_pose')['position'],
            mass = env_obj.get('mass'),
        )
        
        geom = site_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_MESH,
            meshname = f"{obj_name}_mesh",
            quat = env_obj.get('attach_pose')['quaternion'] #  [ 0.5, 0.5, 0.5, 0.5 ]
        )
        
        mesh = self._mj_spec.add_mesh() # initialize the mesh object
        mesh.name = f"{obj_name}_mesh"
        mesh.file = mesh_path
        mesh.scale = env_obj.get('scale')
        
        return
