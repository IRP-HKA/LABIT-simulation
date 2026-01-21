"""
Examples to inspect the robot xml file

Docs:
 - mjtObj: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtobj
 
 - MjModel data structure: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel
"""


import mujoco
import mujoco.viewer
from mujoco import mj_id2name, mjtObj


# model = mujoco.MjModel.from_xml_path('assets/robots/ur5e/ur5e.xml')
model = mujoco.MjModel.from_xml_path('assets/robots/ur5e/scene.xml')
data = mujoco.MjData(model)


def print_object_names(model, obj_type, count, label):
    """Prints the names for objects of a given type from the model."""
    print(f"{label}:")
    for i in range(count):
        # Retrieve the name of the object using its type and id.
        name = mj_id2name(model, obj_type, i)
        if name is not None:
            print(f"  - {name}")
        else:
            print(f"  - [unnamed] (id {i})")

# Print the names for different object types in the model:
print_object_names(model, mjtObj.mjOBJ_BODY, model.nbody, "Bodies")
print_object_names(model, mjtObj.mjOBJ_JOINT, model.njnt, "Joints")
print_object_names(model, mjtObj.mjOBJ_GEOM, model.ngeom, "Geoms")
print_object_names(model, mjtObj.mjOBJ_SITE, model.nsite, "Sites")
print_object_names(model, mjtObj.mjOBJ_ACTUATOR, model.nuser_actuator, "User defined actuators")

# Tendons might not be defined in all models.
if hasattr(model, "ntendon") and model.ntendon > 0:
    print_object_names(model, mjtObj.mjOBJ_TENDON, model.ntendon, "Tendons")
