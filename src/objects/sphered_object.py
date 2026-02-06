import mujoco
import numpy as np
import trimesh
import os
from src.objects.base_object import BaseObject


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
            solref = self._materials[config.get('material')].solref,
            friction = self._materials[config.get('material')].friction,
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
        
        material = self._config.get('material', 'default')
        color = self._config.get('mesh_color')
        if self._config.get('show_spheres', False): group = 2
        else: group = 3

        solref = self._materials[material].solref
        friction = self._materials[material].friction
        condim = config.get('contact').get('condim', 3)
        points = self.FINAL_POINTS - center_in_parent

        # for point, radius in zip(self.FINAL_POINTS, self.FINAL_RADII):
        for i in range(len(points)):
            geom = self.obj_body.add_geom(
                type = mujoco.mjtGeom.mjGEOM_SPHERE,
                group = group, # make invisible in visualizer with group = 3
                condim = condim,
                rgba = color,
                size = [self.FINAL_RADII[i]] * 3,
                pos = points[i], #list(point-center_in_parent),
                mass = self._obj_mass/len(self.FINAL_POINTS),
                solref = solref,
                friction = friction, # sliding friction between the two task objects
            )

        print("loaded sphered object {}".format(config.get('obj_name')))


