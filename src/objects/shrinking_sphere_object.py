"""
Sphere decomposition via the Shrinking Sphere Algorithm.

For each surface sample point p with outward normal n:
  1. Cast a ray from p along -n; get the first interior intersection at distance d.
  2. Initialise sphere: centre c = p + (d/2)*(-n), radius r = d/2.
     The sphere passes through p by construction (|c - p| = r).
  3. Find the closest surface point q to c.  If |c - q| >= r the sphere is
     already inscribed — stop.
  4. Shrink to tangency at q: solve for t such that the sphere centred at
     p + t*(-n) with radius t is tangent to the mesh at q.
     Derivation (|p + t*d - q|^2 = t^2, |d|=1, d = -n):
         t = |p - q|^2 / (2 * dot(q - p, d))
  5. Set r = t, c = p + t*d and repeat from 3.

References
----------
Inui et al. (2016) "Shrinking Sphere: A Parallel Algorithm for Computing
the Thickness of 3D Objects", Computer-Aided Design & Applications,
Vol. 13(2), pp. 199-207.
https://www.cad-journal.net/files/vol_13/CAD_13(2)_2016_199-207.pdf

Inui et al. (2015) "Thickness and clearance visualization based on distance
field of 3D objects", Journal of Computational Design and Engineering,
Vol. 2(3). https://doi.org/10.1016/j.jcde.2015.04.001
"""

import os

import mujoco
import numpy as np
import trimesh
from tqdm import tqdm

from src.objects.base_object import BaseObject


class ShrinkingSphereObject(BaseObject):
    """
    Builds an interior sphere decomposition using the Shrinking Sphere
    Algorithm (Inui et al. 2016). Each sphere is a Maximum Inscribed Sphere
    (MIS) anchored at a surface sample point and grown to the largest radius
    that fits inside the mesh without intersecting its surface.

    Config keys (in addition to BaseObject keys)
    --------------------------------------------
    n_samples      : int   – surface sample points (default 2000)
    max_iterations : int   – shrinking iterations per sphere (default 20)
    min_radius     : float – discard spheres smaller than this (default 0.001)
    coverage_ratio : float – greedy filter: drop sphere j if its centre lies
                             within coverage_ratio * r_i of a larger kept
                             sphere i (default 0.9)
    """

    def __init__(self, mj_spec, config_dict: dict):
        super().__init__(mj_spec, config_dict)

        mesh_path = self._config.get('mesh_path')
        self._cache_path = os.path.splitext(mesh_path)[0] + "_shrinking_spheres.npy"

        if not os.path.exists(self._cache_path):
            mesh = trimesh.load(mesh_path)
            mesh.vertices *= np.array(self._config.get('scale'))
            self._compute_shrinking_spheres(mesh)

        self._load_shrinking_sphere_object(self._config)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _compute_shrinking_spheres(self, mesh: trimesh.Trimesh):
        n_samples      = self._config.get('n_samples', 2000)
        max_iter       = self._config.get('max_iterations', 20)
        min_radius     = self._config.get('min_radius', 0.001)
        coverage_ratio = self._config.get('coverage_ratio', 0.9)

        print(f"[ShrinkingSphere] Sampling {n_samples} surface points ...")
        points, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
        normals = mesh.face_normals[face_ids]          # outward unit normals

        # Offset origins slightly inward so the first cast doesn't
        # self-intersect the source triangle.
        origins    = points - normals * 1e-5
        directions = -normals                          # inward rays

        print("[ShrinkingSphere] Casting initial rays ...")
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        hit_locs, ray_ids, _ = intersector.intersects_location(
            origins, directions, multiple_hits=False
        )

        # Build ray_index -> first hit location map
        ray_to_hit: dict[int, np.ndarray] = {}
        for loc, rid in zip(hit_locs, ray_ids):
            if rid not in ray_to_hit:
                ray_to_hit[rid] = loc

        proximity = trimesh.proximity.ProximityQuery(mesh)

        centers: list[np.ndarray] = []
        radii:   list[float]      = []

        for i, (p, d) in enumerate(tqdm(zip(points, directions), total=len(points), desc="Shrinking spheres")):
            if i not in ray_to_hit:
                continue                            # ray escaped (open mesh / grazing)

            dist = float(np.linalg.norm(ray_to_hit[i] - p))
            if dist < 2.0 * min_radius:
                continue

            r = dist / 2.0
            c = p + d * r

            for _ in range(max_iter):
                q, surface_dist, _ = proximity.on_surface(c[np.newaxis])
                q            = q[0]
                surface_dist = float(surface_dist[0])

                if surface_dist >= r - 1e-8:
                    break                          # sphere is inscribed

                # Shrink: tangency condition at q gives a linear equation in t.
                # t = |p - q|^2 / (2 * dot(q - p, d))
                v     = p - q
                denom = 2.0 * float(np.dot(q - p, d))
                if denom < 1e-12:
                    break                          # degenerate (q behind ray)

                r = float(np.dot(v, v)) / denom
                c = p + d * r

                if r < min_radius:
                    r = 0.0
                    break

            if r >= min_radius:
                centers.append(c)
                radii.append(r)

        if not radii:
            print("[ShrinkingSphere] Warning: no valid spheres found.")
            np.save(self._cache_path,
                    {"positions": np.empty((0, 3)), "radii": np.empty(0)})
            return

        centers_arr = np.array(centers)
        radii_arr   = np.array(radii)

        print(f"[ShrinkingSphere] {len(radii_arr)} raw spheres; filtering ...")
        centers_arr, radii_arr = self._filter_overlapping(
            centers_arr, radii_arr, coverage_ratio
        )
        print(f"[ShrinkingSphere] {len(radii_arr)} spheres retained.")

        np.save(self._cache_path,
                {"positions": centers_arr, "radii": radii_arr})

    @staticmethod
    def _filter_overlapping(
        centers: np.ndarray,
        radii:   np.ndarray,
        coverage_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Greedy largest-first selection.  Sphere j is discarded when its centre
        falls within coverage_ratio * r_i of a larger already-kept sphere i,
        meaning it provides redundant interior coverage.
        """
        order   = np.argsort(radii)[::-1]
        centers = centers[order]
        radii   = radii[order]

        keep = np.ones(len(radii), dtype=bool)
        for i in range(len(radii)):
            if not keep[i]:
                continue
            # Vectorised check against all subsequent candidates
            remaining = np.where(keep[i + 1:])[0] + i + 1
            if len(remaining) == 0:
                break
            dists = np.linalg.norm(centers[remaining] - centers[i], axis=1)
            keep[remaining[dists <= radii[i] * coverage_ratio]] = False

        return centers[keep], radii[keep]

    # ------------------------------------------------------------------
    # MuJoCo loading
    # ------------------------------------------------------------------

    def _load_shrinking_sphere_object(self, config: dict):
        data             = np.load(self._cache_path, allow_pickle=True).item()
        self.FINAL_POINTS = data["positions"]
        self.FINAL_RADII  = data["radii"]

        center_in_parent = (
            np.array([0.0, 0.0, 0.0])
            if config.get('obj_name') in self.objects_to_not_center_cs
            else self.mesh_center
        )

        # Visual-only mesh geom (no collision contribution)
        self.obj_body.add_geom(
            type        = mujoco.mjtGeom.mjGEOM_MESH,
            meshname    = f"{config.get('obj_name')}_mesh",
            pos         = -center_in_parent,
            condim      = 1,
            conaffinity = 0,
            contype     = 0,
            rgba        = config.get('mesh_color', [1, 0, 0, 1]),
            solref      = self._materials[config.get('material')].solref,
            friction    = self._materials[config.get('material')].friction,
        )

        mj_mesh       = self._mj_spec.add_mesh()
        mj_mesh.name  = f"{config.get('obj_name')}_mesh"
        mj_mesh.file  = config.get('mesh_path')
        mj_mesh.scale = config.get('scale')

        material = config.get('material', 'default')
        color    = config.get('mesh_color')
        group    = 2 if config.get('show_spheres', False) else 3
        condim   = config.get('contact', {}).get('condim', 3)
        points   = self.FINAL_POINTS - center_in_parent
        n        = len(self.FINAL_POINTS)

        for i in range(n):
            self.obj_body.add_geom(
                type     = mujoco.mjtGeom.mjGEOM_SPHERE,
                group    = group,
                condim   = condim,
                rgba     = color,
                size     = [float(self.FINAL_RADII[i])] * 3,
                pos      = points[i],
                mass     = self._obj_mass / n,
                solref   = self._materials[material].solref,
                friction = self._materials[material].friction,
            )

        print(f"[ShrinkingSphere] Loaded {n} spheres for '{config.get('obj_name')}'")
