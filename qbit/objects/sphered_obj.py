"""
Sphered-based objects
"""


from typing import Tuple, Literal
import numpy as np

import open3d as o3d

import mujoco
import mujoco.viewer

from qbit.objects.object_base import ObjectBase


class SpheredObj(ObjectBase):

    def __init__(self, mj_spec, obj_body_name, mesh_path, mesh_type):
        super().__init__(mj_spec, obj_body_name, mesh_path, mesh_type)


    def mesh_preprocessing(self,
                           gender: Literal['male', 'female'] = 'female',
                           num_spheres: int = 15000,
                           sphere_radius: float = 0.001,
                           ):
        """
        Preprocess the mesh object.
        Returns:
            - points_on_surface: VERTICES np.ndarray, shape=(num_spheres, 3)
            - face_normals: np.ndarray, shape=(num_spheres, 3)
            - length_z: float
            - origin: np.ndarray, shape=(3,)
        """
        
        print("Start mesh preprocessing...")
        
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        
        # scale and centralize
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)/1000.0)
        mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices) - mesh.get_center())
        
        length_z = mesh.get_max_bound()[2] - mesh.get_min_bound()[2]
        
        if gender == "female":
            origin = mesh.get_center() + np.abs(mesh.get_min_bound()[2])
        elif gender == "male":
            origin = mesh.get_center() + np.abs(mesh.get_max_bound()[2])
        
        # sample points on the surface
        pcd = mesh.sample_points_uniformly(number_of_points=num_spheres)
        dpcd = pcd.voxel_down_sample(voxel_size=sphere_radius*1.5)
        
        face_normals = np.array(dpcd.normals)
        points_on_surface = np.array(dpcd.points)

        # TODO: rename
        self.mesh = mesh
        self.points_on_surface = points_on_surface
        self.face_normals = face_normals
        self.mesh_origin = origin
        self.sphere_radius = sphere_radius
        self.gender = gender        

        return points_on_surface, face_normals, length_z, origin

    
    def attach_spheres_to_body(self,
                               parent_body, #str,
                               every_nth_sphere: int = 1,
                               friction: float = 0.7, # steel, foam
                               male_tolerance: Tuple [float, float] = [0.000004, 0.000009],
                               female_tolerance: Tuple [float, float] = [-0.0000, -0.000], # [0.0, 0.000012],
                            #    female_tolerance: Tuple [float, float] = [-0.0000, -0.0002], # [0.0, 0.000012],
                               male_material_density: int = 7850, # steel
                               female_material_density: int = 7850 # foam,
                               ):
        print("Attaching spheres to the body...")
        print(f"Number of spheres: {len(self.points_on_surface)}")
        # parent_body = self._mj_spec.find_body(parent_body_name)

        for i, v in enumerate(self.points_on_surface):
            
            if i % every_nth_sphere == 0:
                
                # if v[1] > 0.053:
                #     continue
                    
                if self.gender == "female":
                    geom = parent_body.add_geom()
                    geom.name = parent_body.name + "_" + str(i)
                    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    geom.condim = 6

                    normal = self.face_normals[i] # mesh.vertex_normals[i]

                    # print(np.linalg.norm(normal))
                    normal = normal / np.linalg.norm(normal)
                    
                    tol_low = female_tolerance[0]
                    tol_high = female_tolerance[1]
                    offset_by_tolerance = np.random.uniform(low=tol_low, high=tol_high)
                    
                    geom.pos = v - normal * self.sphere_radius - normal * offset_by_tolerance/2 
                    geom.rgba = [0, 1, 0, 1]
                    geom.size = [self.sphere_radius, self.sphere_radius, self.sphere_radius]
                    
                    # geom.rgba = np.random.rand(3).tolist() + [1.0]
                    # geom.rgba = [0, 1, 0, 1]
                    geom.friction[0] = friction
                    geom.density = (male_material_density + female_material_density)/2
        # print( self.mesh_origin)
        print("parent_body.pos", parent_body.pos)
        return


if __name__ == "__main__":
    obj = SpheredObj()
