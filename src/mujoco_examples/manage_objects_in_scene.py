"""
Manage objects in the Mujoco Model 
 - add site object
 - add moving object
 - add mesh object
"""

import mujoco
import mujoco.viewer
import time

xml_path = '/workspace/qacbi/assets/robots/ur5e/scene.xml'


### Editing the model
spec = mujoco.MjSpec()
spec.from_file(xml_path)


### Create a new world object (Site object)
# Site is essentially a marker or reference point that you attach to body
# It can be used for the following purposes:
# - reference frames
# - Visualization and debugging
# - Sensor attachment
# - Collision detection
site_body = spec.worldbody.add_body(
    name = 'site_body',
    pos = [0, 0, 1],
    mass = 1.0,
)
site = site_body.add_site(
    name = 'test_site',
    type = mujoco.mjtGeom.mjGEOM_SPHERE,
    size = [0.1, 0.1, 0.1],
)

### Moving body
moving_body = spec.worldbody.add_body(
    name = 'moving_body',
    pos = [0, 0.5, 1],
    mass = 1.0,
    ipos = [0.1, 0.1, 0.1],
    inertia = [1, 1, 1],
)
# # add geom
i_geom = moving_body.add_geom(
    type = mujoco.mjtGeom.mjGEOM_SPHERE,
    size = [0.1, 0.1, 0.1],
    rgba = [1, 0, 0, 1],
    mass = 1
)
# add joint
joint = moving_body.add_joint(
    name = 'interactive_joint',
    type = mujoco.mjtJoint.mjJNT_FREE,
    # align=True
)
# joint = moving_body.add_freejoint()


### Attach mesh object to the end effector ###
eef_body = spec.find_body('wrist_3_link').add_body()
geom = eef_body.add_geom(
    type = mujoco.mjtGeom.mjGEOM_MESH,
    meshname = 'test_mesh',
    size = [0.1, 0.1, 0.1]
)
# add mesh to the assets
print("Meshdir: ", spec.meshdir)
cube = """
      v -1 -1  1
      v  1 -1  1
      v -1  1  1
      v  1  1  1
      v -1  1 -1
      v  1  1 -1
      v -1 -1 -1
      v  1 -1 -1"""
mesh = spec.add_mesh() # initialize the mesh object
mesh.name = 'test_mesh'
mesh.file = 'cube.obj'
mesh.scale = [0.05, 0.05, 0.05]
# push it to the added assets
added_assets = {'cube.obj': cube}

# Compile the model
model =  spec.compile(added_assets)
# print(spec.to_xml())


### Inspection ###

# print(model.body_parentid)
# print(model.body_jntnum)
# print(model.body_pos)

data = mujoco.MjData(model)
ctrl = data.ctrl.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        
        data.ctrl[0] += 0.003
        
        mujoco.mj_step(model, data)
        # print(data.qpos)
        viewer.sync()

        time.sleep(0.001)
        # print(model.body_pos)
        # print(data.xpos)
        # print("-" *30)

        # get id of body via name
        id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "peg_body")
        peg_pos = data.xpos[id, :]
