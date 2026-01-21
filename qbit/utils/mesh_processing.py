"""
    Utils for mesh processing    
    - CoACD: https://github.com/SarahWeiii/CoACD
"""
import os
import glob
import numpy as np
import trimesh
import coacd
import subprocess


class MeshObjects:
    
    
    def __init__(self, obj_path):
        
        self._obj_path = obj_path
        self._decomposed_mesh_dir_name = obj_path.split(os.sep)[-1][:-4] + "_decomposed"
        
        self._decomposed_mesh_dir = os.path.join(
            os.path.dirname(os.path.abspath(self._obj_path)),
                            self._decomposed_mesh_dir_name)

    def show_decomposed_mesh(self):

        mesh_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(self._obj_path)), 
                                            self._decomposed_mesh_dir_name, "*.obj"))        
        scene = trimesh.Scene()
        for i, f in enumerate(mesh_files):
            
            mesh = trimesh.load(f, process=False)
                        
            if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
                mesh.fix_normals()
            mesh.visual.material = trimesh.visual.material.SimpleMaterial(
                diffuse=[1.0, 0.0, 0.0, 1.0]  # Red diffuse color (values in 0-1)
            )                
            mesh.visual = trimesh.visual.ColorVisuals(mesh)
                
            mesh.visual.vertex_colors = np.tile([255, 0, 0, 255], (mesh.vertices.shape[0], 1))
            mesh.visual.face_colors[:, :4] = (np.random.rand(4) * 255).astype(np.uint8)
            
            scene.add_geometry(mesh, node_name=f"part_{i}")

        scene.show(flags={'wireframe':True})

    def show_mesh(self):
        scene = trimesh.Scene()
        mesh = trimesh.load(self._obj_path)
        mesh.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(mesh)
        
        scene.show(flags={'wireframe':True})

    def decomposition_with_vhacd(self):
        """
        maxConvexHulls                    64
        resolution                        400000
        minimumVolumePercentErrorAllowed  1.0
        maxRecursionDepth                 10
        shrinkWrap                        True
        fillMode                          "flood"
        maxNumVerticesPerCH               64
        asyncACD                          True
        minEdgeLength                     2
        findBestPlane                     False
        """

        self._decomposed_mesh_dir += "_vhacd"

        if not os.path.exists(self._decomposed_mesh_dir):
            os.makedirs(self._decomposed_mesh_dir)

            mesh = trimesh.load(self._obj_path)

            decomposed_meshes = trimesh.decomposition.convex_decomposition(mesh, resolution=10000000, maxConvexHulls=100, minimumVolumePercentErrorAllowed=0.1)
            
            for i, p in enumerate(decomposed_meshes):
                m = trimesh.Trimesh(p['vertices'], p['faces'])
                m.visual.face_colors = (np.random.rand(3) * 255).astype(np.uint8)
                m.visual.vertex_colors = (np.random.rand(3) * 255).astype(np.uint8)
                m.export(f"{self._decomposed_mesh_dir}/part_{i}.obj")
            print(f"[MESH PROCESSING] Decomposed meshes ({i} parts) are saved in {self._decomposed_mesh_dir}")

    def decomposition_with_coacd(self, threshold=0.02):
        
        self._decomposed_mesh_dir += "_coacd"

        if not os.path.exists(self._decomposed_mesh_dir):
            os.makedirs(self._decomposed_mesh_dir)
            
            mesh = trimesh.load(self._obj_path)
            mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            parts = coacd.run_coacd(mesh, threshold=threshold, preprocess_resolution=60, resolution=4000, mcts_max_depth=5, mcts_iterations=1000, max_convex_hull=128)
            
            for i, p in enumerate(parts):
                mesh = trimesh.Trimesh(p[0], p[1])
                mesh.visual.face_colors = trimesh.visual.color.random_color()
                mesh.export(f"{self._decomposed_mesh_dir}/part_{i}.obj")
            print(f"[MESH PROCESSING] Decomposed meshes ({i} parts) are saved in {self._decomposed_mesh_dir}")
        
    def convert_stl_to_msh(self,):
        """
        Converts an STL file to a tetrahedral GMSH mesh using fTetWild and GMSH.
        
        Parameters:
        - ftetwild_path: Path to the fTetWild executable
        - output_msh_path: Desired output .msh file path
        """
        ftetwild_path = "/workspace/qbit/objects/fTetWild/build/FloatTetwild_bin"
        self.output_msh_path = self._obj_path[:-4] + ".msh"

        if not os.path.exists(self.output_msh_path):
            assert os.path.isfile(self._obj_path), f"[MESH PROCESSING] Input STL file not found: {self._obj_path}"

            # Step 1: Run fTetWild
            print(f"[MESH PROCESSING] Running fTetWild on {self._obj_path}...")
            try:
                subprocess.run(
                    [ftetwild_path, "-i", self._obj_path, "-o", self.output_msh_path, "--coarsen", "--no-binary"],
                    check=True
                )
            except subprocess.CalledProcessError:
                print("[MESH PROCESSING] Error: fTetWild failed to run.")
                return False

            print("[MESH PROCESSING] Conversion completed successfully. \n")
            return True      

        print(f"[MESH PROCESSING] {self.output_msh_path} already exists. Skipping converting from stl to msh. \n")



if __name__ == "__main__":
    

    obj_path = "/workspace/qbit/assets/task_env/primitives/box_30.831x11.208x30.872/box_30.831x11.208x30.872_female.stl"
   
    # obj_path = "/workspace/qacbi/assets/task_env/plugs/dsub_25_female.stl"
    
    mo = MeshObjects(obj_path)
    mo.decomposition_with_coacd(threshold=0.01)
    