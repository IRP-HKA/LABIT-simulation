"""
Task object base class
"""

import gmsh
import os
import glob
from typing import Tuple, Literal
from loguru import logger
import xml.etree.ElementTree as ET

import numpy as np

import trimesh
from scipy.spatial.transform import Rotation as R

import mujoco

from qbit.utils.tf_utils import T
from qbit.utils.mujoco_utils import convert_quat_to_xyzw
from qbit.utils.mesh_processing import MeshObjects
from qbit.utils.mujoco_material_definitions import MATERIALS



class BaseObject:
    def __init__(self, 
                 mj_spec, 
                 config_dict: dict):
        
        self._mj_spec = mj_spec
        self._config = config_dict
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
        self._obj_mass = self._obj_volume * MATERIALS[config["material"]].density

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
            self.obj_body = self._mj_spec.find_body(parent_body_name).add_body(
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
                pos = self._mj_spec.find_body(parent_body_name).pos
                quat = self._mj_spec.find_body(parent_body_name).quat
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
                )
                self.obj_body.add_freejoint()
            else:
                self.obj_body = self._mj_spec.find_body(parent_body_name).add_body(
                    name = f"{config.get('obj_name')}_body",
                    pos = config.get('attach_pose')['position'] + center_in_parent,
                    quat = config.get('attach_pose')['quaternion'],
                    )
            
            

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
                density = MATERIALS[config.get('material')].density,
                solref = MATERIALS[config.get('material')].solref,
                friction = MATERIALS[config.get('material')].friction,
            )
            mesh = self._mj_spec.add_mesh()
            mesh.name = f"{config.get('obj_name')}_mesh_{i}"
            mesh.file = f
            mesh.scale = config.get('scale')

        print("loaded mesh files")


class MeshObject(BaseObject):
    
    def __init__(self,
                 mj_spec,
                 config_dict: dict):
        super(MeshObject, self).__init__(mj_spec, config_dict)
        self.load_mesh_object(self._config)
        
    def load_mesh_object(self, config):
        
        if config.get('obj_name') in ["plate_benchmark", "housing_bottom", "housing_top", "housing_middle", "rotor_printed_part", "pcb"]: center_in_parent = np.array([0,0,0])
        else: center_in_parent = self.mesh_center

        # load the mesh files
        geom = self.obj_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_MESH,
            meshname = f"{config.get('obj_name')}_mesh",
            pos = -center_in_parent,
            condim = config.get('contact').get('condim', 3),
            rgba = config.get('mesh_color', [1, 0, 0, 1]),
            mass = self._obj_mass, #config.get('mass'),
            solref = MATERIALS[config.get('material')].solref,
            friction = MATERIALS[config.get('material')].friction,
        )

        mesh = self._mj_spec.add_mesh()
        mesh.name = f"{config.get('obj_name')}_mesh"
        mesh.file = config.get('mesh_path')
        mesh.scale = config.get('scale')

        print(f"loaded object {config.get('obj_name')}")


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
            solref = MATERIALS[config.get('material')].solref,
            friction = MATERIALS[config.get('material')].friction,
        )
        geom.plugin.instance_name = "sdf1"
        geom.plugin.active = 1
        
        mesh = self._mj_spec.add_mesh()
        mesh.name = f"{config.get('obj_name')}_mesh"
        mesh.file = config.get('mesh_path')
        mesh.scale = config.get('scale')
        mesh.plugin.instance_name = "sdf1"

        print(f"loaded object {config.get('obj_name')}")


class BuildInObject:
    """
    Add the built-in object (primitive shapes) to the MuJoCo model
    """

    def __init__(self,
                 mj_spec):
        self._mj_spec = mj_spec


    def add_box(self, 
                pose: T,
                box_size: Tuple[float, float, float],
                obj_name = 'box-1'):
        
        body = self._mj_spec.worldbody.add_body(
            name = obj_name,
            pos = pose.translation,
            quat = convert_quat_to_xyzw(pose.quaternion),
            mass = 1.0,
        )
        geom = body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_BOX,
            size = box_size,
            density = 1000,
            rgba = [0, 0, 1, 0.5],
            condim = 3,
        )
        geom.friction[0] = 0.1
        return


class FlexcompObject(BaseObject):
    def __init__(self,
                mj_spec,
                config_dict: dict):
        super(FlexcompObject, self).__init__(mj_spec, config_dict)
        
        _mp = MeshObjects(obj_path=self._config.get('mesh_path'))
        _mp.convert_stl_to_msh()

        self.msh_path = _mp.output_msh_path
        # self.msh_path = self._config.get('mesh_path')[:-4]+"_processed.stl"

        self.nodes = self.parse_nodes_from_msh(file_path=self.msh_path)
        self.load_flexcomp_object(self._config)
    
    def load_flexcomp_object(self, config):
        
        # compile spec
        self._mj_model = self._mj_spec.compile()

        # save to xml
        xmlstring = self._mj_spec.to_xml()
        root = ET.fromstring(xmlstring)

        # parse flexcomp to xml
        element_body = root.findall(".//*[@name='" + self.obj_body.name + "']")[0] # unique body name

        element_flexcomp = ET.SubElement(element_body, "flexcomp")
        element_flexcomp.set("rgba", " ".join([str(c) for c in config['mesh_color']]))
        element_flexcomp.set("scale", " ".join([str(c) for c in config['scale']]))
        element_flexcomp.set("radius", "0.00001")
        element_flexcomp.set("dim", "3")
        element_flexcomp.set("file", self.msh_path)
        element_flexcomp.set("mass", str(self._obj_mass))
        element_flexcomp.set("name", self.obj_body.name)
        element_flexcomp.set("type", "gmsh")

        element_contact = ET.SubElement(element_flexcomp, "contact")
        element_contact.set("condim", "1")
        element_contact.set("selfcollide", "none") # bvh
        element_contact.set("internal", "false")
        # element_contact.set("activelayers", "1")
        element_contact.set("solimp", "0.95 0.99 0.001 0.5 2") # 0.0001
        element_contact.set("solref", " ".join([str(c) for c in MATERIALS[config["material"]].solref])) # "0.01 1"

        element_edge = ET.SubElement(element_flexcomp, "edge")
        element_edge.set("damping", "0.5")
        element_edge.set("equality", "true")
        # element_edge.set("solimp", "0.95 0.99 0.001 0.5 2") # 0.0001
        # element_edge.set("solref", " ".join([str(c) for c in MATERIALS[config["material"]]["solref"]])) # "0.01 1"

        element_plugin = ET.SubElement(element_flexcomp, "plugin")
        element_plugin.set("plugin", "mujoco.elasticity.solid")

        element_config_0 = ET.SubElement(element_plugin, "config")
        element_config_0.set("key", "young")   
        element_config_0.set("value", str(MATERIALS[config["material"]].young))

        element_config_1 = ET.SubElement(element_plugin, "config")
        element_config_1.set("key", "poisson")   
        element_config_1.set("value", str(MATERIALS[config["material"]].poisson))

        # pin all the points which are at the bottom of the flexobject
        z_threshold = np.array(self.nodes)[:,3].min()
        pinned_node_indices = [i for i, x, y, z in self.nodes if z <= z_threshold]
        element_pin = ET.SubElement(element_flexcomp, "pin")
        element_pin.set("id", " ".join([str(c) for c in pinned_node_indices]))

        
        new_xmlstring = ET.tostring(root)

        # load spec from updated xml string
        self._mj_spec.from_string(new_xmlstring)
        
        print(f"loaded object {config.get('obj_name')}")

    def parse_nodes_from_msh(self, file_path):
        nodes = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            in_nodes_section = False
            for i, line in enumerate(lines):
                if "$Nodes" in line:
                    num_nodes = int(lines[i+1])
                    for j in range(num_nodes):
                        parts = lines[i+2+j].strip().split()
                        if len(parts) >= 4:
                            _, x, y, z = parts
                            nodes.append((j, float(x), float(y), float(z)))
                    break
        return nodes


class SpheredObject(BaseObject):
    """
    Todo: git clone SpheredDecomposition git project and use the class to load and decomp etc.
    right now only loading to test.

    """
    def __init__(self,
                 mj_spec,
                 config_dict):
        super(SpheredObject, self).__init__(mj_spec, config_dict)

        self._sphered_object_dir = self._config.get('mesh_path').replace(".stl", "_sphered.npy")

        if not os.path.exists(self._sphered_object_dir):
            self.sphere_packing_sdf(mesh=trimesh.load(self._config.get('mesh_path')),
                                    radius=self._config.get('radius_spheres', 0.001), bboxes=self._config.get('bboxes_fine', []))

        self.load_sphered_object(config=self._config)


    def sphere_packing_sdf(self, mesh, radius, bboxes=[]):
        """
        Generate sphere packing using SDF method
        Args:
            mesh: trimesh object
            radius: float, radius of the spheres
            bboxes: list of bounding boxes for fine sampling, each bbox is a tuple of (min, max) coordinates
        Returns:
            saves the sphere packing to self._sphered_object_dir as a .npy file
        """
        mesh.vertices *= np.array(self._config.get('scale'))
        bounds = mesh.bounds
        radius_fine = radius / 4
        
        print("[sphere_packing_sdf] Generating sample points...")
        # coarse sampling
        x = np.arange(bounds[0][0]+radius, bounds[1][0]+radius, radius*2)
        y = np.arange(bounds[0][1]+radius, bounds[1][1]+radius, radius*2)
        z = np.arange(bounds[0][2]+radius, bounds[1][2]+radius, radius*2)

        X,Y,Z = np.meshgrid(x,y,z)
        sample_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        print("[sphere_packing_sdf] Computing signed distances...")
        signed_distance = trimesh.proximity.signed_distance(mesh, sample_points)
        inside_points_coarse = sample_points[(signed_distance >= radius) & (signed_distance < 3*radius)]

        N = len(inside_points_coarse)
        radii_coarse = np.ones((N))*radius

        print("[sphere_packing_sdf] Generating fine sample points in bboxes...")
        for i,bbox in enumerate(bboxes):
            bbox_min = bbox[0]
            bbox_max = bbox[1]

            # fine sampling
            x = np.arange(bbox_min[0], bbox_max[0], radius_fine*2)
            y = np.arange(bbox_min[1], bbox_max[1], radius_fine*2)
            z = np.arange(bbox_min[2], bbox_max[2], radius_fine*2)

            X,Y,Z = np.meshgrid(x,y,z)
            sample_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

            signed_distance = trimesh.proximity.signed_distance(mesh, sample_points)
            inside_points_fine = sample_points[(signed_distance >= radius_fine) & (signed_distance < 3*radius_fine)]

            N = len(inside_points_fine)
            radii_fine = np.ones((N))*radius_fine

            inside_points_coarse = np.vstack((inside_points_coarse, inside_points_fine))
            radii_coarse = np.hstack((radii_coarse, radii_fine))   
            print("[sphere_packing_sdf] Added " + str(N) + " fine points in bbox " + str(i))
        
        np.save(self._sphered_object_dir, {"radii": radii_coarse, "positions": inside_points_coarse})


    def load_sphered_object(self, config):
        file = self._sphered_object_dir
        scale = np.array(config.get('scale'))

        if config.get('obj_name') in self.objects_to_not_center_cs: center_in_parent = np.array([0,0,0])
        else: center_in_parent = self.mesh_center

        # add mesh for visual
        geom = self.obj_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_MESH,
            meshname = f"{config.get('obj_name')}_mesh",
            pos = -center_in_parent,
            condim = 1,
            conaffinity = 0, # remove collision
            contype = 0,    # remove collision
            rgba = config.get('mesh_color', [1, 0, 0, 1]),
            solref = MATERIALS[config.get('material')].solref,
            friction = MATERIALS[config.get('material')].friction,
        )

        mesh = self._mj_spec.add_mesh()
        mesh.name = f"{config.get('obj_name')}_mesh"
        mesh.file = config.get('mesh_path')
        mesh.scale = config.get('scale')

        if file.endswith(".npy"):
            decomposed_mesh = np.load(file, allow_pickle=True)
            self.FINAL_POINTS = decomposed_mesh.item()["positions"]
            self.FINAL_RADII = decomposed_mesh.item()["radii"]
            # FINAL_COLORS = decomposed_mesh.item()["colors"]
        # elif file.endswith(".csv"): # xProtosphere support
        #     # data is of shape (n,4): [x, y, z, radius], n being number of spheres
        #     # skip header in .csv with skiprows=1
        #     data = np.loadtxt(file, delimiter=',', skiprows=1)
        #     scale_xproto = np.max(self.mesh_extents/(np.max(data[:, :3])-np.min(data[:, :3])))
        #     self.FINAL_POINTS = data[:, :3][::5] * scale_xproto * scale + self.mesh_center * scale
        #     self.FINAL_RADII = data[:, 3][::5] * scale_xproto * scale[0]
        else:
            raise NotImplementedError("Only .npy sphere decomposition files are supported for now.")
        
        for point, radius in zip(self.FINAL_POINTS, self.FINAL_RADII):
            
            material = self._config.get('material', 'default')
            color = self._config.get('mesh_color')
            if self._config.get('show_spheres', False): group = 2
            else: group = 3

            geom = self.obj_body.add_geom(
                type = mujoco.mjtGeom.mjGEOM_SPHERE,
                group = group, # make invisible in visualizer with group = 3
                condim = config.get('contact').get('condim', 3),
                rgba = color,
                size = [radius]*3,
                pos = list(point-center_in_parent),
                mass = self._obj_mass/len(self.FINAL_POINTS),
                solref = MATERIALS[material].solref,
                friction = MATERIALS[material].friction, # sliding friction between the two task objects
            )

        print("loaded sphered object {}".format(config.get('obj_name')))



if __name__ == "__main__":
    # mesh_stl_path = "/workspace/qbit/assets/task_env/primitives/box_5.013x20.853x5.204/box_5.013x20.853x5.204_male.stl"
    # mesh_gmsh_path = mesh_stl_path[:-3] + "msh"

    # Load the sphered object file
    sphered_file = "qbit/assets/task_env/labit_benchmark/tube_clamp_sphered.npy"
    decomposed_mesh = np.load(sphered_file, allow_pickle=True)
    positions = decomposed_mesh.item()["positions"]
    radii = decomposed_mesh.item()["radii"]
    
    print("Number of spheres:", len(positions))
    print("Radius of first sphere:", radii[0])

    # Create a simple visualization using trimesh
    spheres = []
    for pos, rad in zip(positions, radii):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=rad)
        sphere.apply_translation(pos)
        spheres.append(sphere)

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
                  [[0.085, 0.06, 0.015], [0.095, 0.07, 0.03]], # plug outside
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