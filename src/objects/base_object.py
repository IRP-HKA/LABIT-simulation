"""
Task object base class
"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import mujoco
from src.utils.tf_utils import T
from src.utils.mujoco_material_definitions import MATERIALS


class BaseObject:
    def __init__(self, 
                 mj_spec, 
                 config_dict: dict):
        
        self._mj_spec = mj_spec
        self._config = config_dict
        self._materials = MATERIALS
        self.objects_to_not_center_cs = ["plate_benchmark", 
                                         "housing_bottom", 
                                         "housing_top", 
                                         "housing_middle", 
                                         "rotor_printed_part", 
                                         "pcb", 
                                         "plug_inside_loose_1", 
                                         "plug_inside_loose_2",
                                         "plug_inside_fixed_1",
                                         "plug_inside_fixed_2"]

        if self._config.get('mesh_path'):
            self.start_position_hole, self.insertion_depth = self.get_hole_pose_depth(self._config)
        self.attach_body(config=self._config)

    def get_hole_pose_depth(self, config):
        self.mesh_path = config.get('mesh_path')
        
        meshfile = self.mesh_path

        mesh = trimesh.load_mesh(meshfile)

        mesh.vertices *= np.array(config.get('scale'))
        self.mesh_extents = mesh.extents
        self.mesh_center = mesh.centroid

        self._obj_volume = mesh.volume
        self._obj_mass = self._obj_volume * self._materials[config["material"]].density

        quat = config["attach_pose"]["quaternion"]
        quat = [quat[1], quat[2], quat[3], quat[0]]
        rotation_matrix = R.from_quat(quat).as_matrix()

        # Create a 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix

        # Apply the rotation
        mesh.apply_transform(transform)

        insertion_depth = mesh.extents[2]
        start_position_hole = np.array(config.get('attach_pose')['position']) + np.array([0, 0, insertion_depth/2 + 0.0])
        
        return start_position_hole, insertion_depth
    
    def attach_body(self, config):
        parent_body_name = config.get('attach_body')

        if config.get('mesh_type') == "none":
            self.obj_body = self._mj_spec.body(parent_body_name).add_body(
                name = f"{config.get('obj_name')}_body",
                pos = config.get('attach_pose')['position'],
                quat = config.get('attach_pose')['quaternion'],
                )
            return

        quat = self._config.get("attach_pose")["quaternion"]  # parent->mesh rotation
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        R_parent_mesh = R.from_quat(quat).as_matrix()
        center_in_parent = R_parent_mesh @ self.mesh_center

        if config.get('obj_name') in self.objects_to_not_center_cs: center_in_parent = np.array([0,0,0])
        if self._config.get('attach_body') == 'world':
            self.obj_body = self._mj_spec.worldbody.add_body(
                name = f"{config['obj_name']}_body",
                pos = self.start_position_hole + center_in_parent,
                quat = config.get('attach_pose')['quaternion'],
                )
        else:
            # if we want to have free moving object, we need to attach to world in order to add a freejoint.
            # If the body should have a free joint, but its parent body is not world, we attach to world
            # given the relative pose of parent body and the pose of the parent body in world.
            
            if config.get('joint') == 'free': 
                pos = self._mj_spec.body(parent_body_name).pos
                quat = self._mj_spec.body(parent_body_name).quat
                quat = np.array([quat[1], quat[2], quat[3], quat[0]])

                quat_ = config.get('attach_pose')['quaternion']
                quat_ = np.array([quat_[1], quat_[2], quat_[3], quat_[0]])

                parent_pose = T(translation=pos, quaternion=quat)._matrix
                
                attach_pose = T(translation=config.get('attach_pose')['position'] + center_in_parent, quaternion=quat_)._matrix
                attach_pose_in_world = T.from_matrix(parent_pose @ attach_pose)
                posquat_world = attach_pose_in_world.get_pos_quat_list(quat_format="wxyz")

                self.obj_body = self._mj_spec.worldbody.add_body(
                    name = f"{config['obj_name']}_body",
                    pos = posquat_world[:3],
                    quat = posquat_world[3:],
                    sleep = mujoco.mjtSleepPolicy.mjSLEEP_INIT if config.get('sleep', True) else mujoco.mjtSleepPolicy.mjSLEEP_AUTO
                )
                self.obj_body.add_freejoint()
            else:
                self.obj_body = self._mj_spec.body(parent_body_name).add_body(
                    name = f"{config.get('obj_name')}_body",
                    pos = config.get('attach_pose')['position'] + center_in_parent,
                    quat = config.get('attach_pose')['quaternion'],
                    )
            
            
            
if __name__ == "__main__":
    # mesh_stl_path = "/workspace/qbit/assets/task_env/primitives/box_5.013x20.853x5.204/box_5.013x20.853x5.204_male.stl"
    # mesh_gmsh_path = mesh_stl_path[:-3] + "msh"

    # Load the sphered object file
    sphered_file = "qbit/assets/task_env/labit_benchmark/parts_supply_sphered.npy"
    mesh_file = "qbit/assets/task_env/labit_benchmark/parts_supply.stl"

    mesh = trimesh.load_mesh(mesh_file)
    mesh.vertices *= np.array([0.001, 0.001, 0.001])
    decomposed_mesh = np.load(sphered_file, allow_pickle=True)
    positions = decomposed_mesh.item()["positions"]
    radii = decomposed_mesh.item()["radii"]
    
    print("Number of spheres:", len(positions))
    print("Radius of first sphere:", radii[0])

    # Create a simple visualization using trimesh
    spheres = []
    i = 0
    for pos, rad in zip(positions, radii):
        if i % 1 == 0:
            if rad < 0.001:
                sphere = trimesh.creation.icosphere(subdivisions=1, radius=rad, face_colors=[0, 0, 255])
            else:
                sphere = trimesh.creation.icosphere(subdivisions=1, radius=rad)
            sphere.apply_translation(pos)
            spheres.append(sphere)
        i += 1

    bboxes = [[[0.0475, 0.00475, 0.015], [0.0625, 0.0195, 0.03]], #tube
                  [[0.08, 0.003, 0.015], [0.09, 0.013, 0.03]],        # pin
                  [[0.105, 0.003, 0.015], [0.115, 0.013, 0.03]],      # pin
                  [[0.13, 0.003, 0.015], [0.14, 0.013, 0.03]],        # screw
                  [[0.155, 0.003, 0.015], [0.165, 0.013, 0.03]],      # screw
                  [[0.18, 0.003, 0.015], [0.19, 0.013, 0.03]],        # screw
                  [[0.007, 0.05, 0.015],[0.017, 0.06, 0.03]],         # pin
                  [[0.007, 0.02, 0.015],[0.017, 0.03, 0.03]],         # pin
                  [[0.08, 0.038, 0.015], [0.09, 0.048, 0.03]],        # pin
                  [[0.105, 0.038, 0.015], [0.115, 0.048, 0.03]],      # pin
                  [[0.13, 0.038, 0.015], [0.14, 0.048, 0.03]],        # screw
                  [[0.155, 0.038, 0.015], [0.165, 0.048, 0.03]],      # screw
                  [[0.04325, 0.06291, 0.0125], [0.05325, 0.07291, 0.0275]], # plug inside
                  [[0.0825, 0.0575, 0.015], [0.0975, 0.0725, 0.03]], # plug outside
                  [[0.12985, 0.06291, 0.0125], [0.13985, 0.07291, 0.0275]], # plug inside
                  ]
    
    # for bbox in bboxes:
    #     bbox_min = bbox[0]
    #     bbox_max = bbox[1]
    #     box = trimesh.creation.box(extents=np.array(bbox_max)-np.array(bbox_min))
    #     box.apply_translation((np.array(bbox_min)+np.array(bbox_max))/2)
    #     box.visual.face_colors = [255, 0, 0, 100]  # Red color with some transparency
    #     spheres.append(box)

    # Combine all spheres into one mesh
    combined = trimesh.util.concatenate(spheres)
    combined.show()