import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr
from qbit.sim_envs.mujoco_env_base import MujocoEnvBase


friction_list = {
            "steel": {
                "steel": 0.4,
                "plastic": 0.2,
                "wood": 0.5,
                "rubber": 1.0,
                "feather": 0.2,
            },
            "plastic": {
                "steel": 0.2,
                "plastic": 0.25,
                "wood": 0.3,
                "rubber": 0.6,
                "feather": 0.2,
            },
            "wood": {
                "steel": 0.5,
                "plastic": 0.3,
                "wood": 0.5,
                "rubber": 0.9,
                "feather": 0.2,
            },
            "rubber": {
                "steel": 0.2,
                "plastic": 0.6,
                "wood": 0.9,
                "rubber": 1.8,
                "feather": 0.2,
            },
            "feather": {
                "steel": 0.2,
                "plastic": 0.2,
                "wood": 0.2,
                "rubber": 0.2,
                "feather": 0.2,
            },
        }

material_list = {
    "steel": {
        "solref": [-500, -0.1], #-0.1
        "density": 7850, 
    },
    "plastic": {
        "solref": [-10.0, -0.1], #-0.01
        "density": 1190, 
    },
    "wood": {
        "solref": [-5.0, -0.1], #-0.01
        "density": 700, 
    },
    "rubber": {
        "solref": [-1.0, -0.1],
        "density": 920, 
    },
    "feather": {
        "solref": [-1.0, -0.1],
        "density": 1,
    },
}


class DataRecording():
    def __init__(self, task_env_config_path):
        self.init()

        self.config = MujocoEnvBase.parse_qbit_config_yaml(task_env_config_path)

        # for PIPE
        self.object_type = self.config["task_objects"][0]["obj_type"]
        self.object_name = self.config["task_objects"][0]["obj_name"]
        self.material_male = self.config["task_objects"][0]["material"]
        self.material_female = self.config["task_objects"][1]["material"]
        self.friction = friction_list[self.material_male][self.material_female]

        self.RESULT_DIR = os.path.join("/workspace/examples/experiment_results/position_based/", self.config["data_recording"]["save_folder"])
        self.RESULT_DIR = os.path.join(self.RESULT_DIR, self.object_name[:-5])

        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

    def init(self):
        self.timestamp = []
        self.eef_fts = []
        self.eef_pos = []
        self.eef_qua = []
        self.joint_states = []
    
    def record(self, timestamp, eef_fts, eef_pos, eef_qua, joint_states):
        self.timestamp.append(timestamp)
        self.eef_fts.append(eef_fts)
        self.eef_pos.append(eef_pos)
        self.eef_qua.append(eef_qua)
        self.joint_states.append(joint_states)

    def save(self,):
        self.filename = self.object_type + "_" + self.object_name + "_" + self.material_male + "_" + self.material_female
        self.savepath = os.path.join(self.RESULT_DIR, self.filename)

        self.timestamp = np.array(self.timestamp)
        self.eef_fts = np.array(self.eef_fts)
        self.eef_pos = np.array(self.eef_pos)
        self.eef_qua = np.array(self.eef_qua)
        self.joint_states = np.array(self.joint_states)

        # np.savez(self.savepath + ".npz",
        #          timestamp = self.timestamp,
        #          eef_fts = self.eef_fts,
        #          eef_pos = self.eef_pos,
        #          eef_qua = self.eef_qua,
        #          joint_states = self.joint_states,
        #          object_type = self.object_type,
        #          material_male = self.material_male,
        #          material_female = self.material_female)
        
        position_array = xr.DataArray(self.eef_pos, coords=[self.timestamp, ["x", "y", "z"]], dims=["time", "axis"], attrs=dict(units="meter"))
        orientation_array = xr.DataArray(self.eef_qua, coords=[self.timestamp, ["x", "y", "z", "w"]], dims=["time", "axis"], attrs=dict(units="radian"))
        force_array = xr.DataArray(self.eef_fts[:,:3], coords=[self.timestamp, ["x", "y", "z"]], dims=["time", "axis"], attrs=dict(units="newton"))
        torque_array = xr.DataArray(self.eef_fts[:,3:], coords=[self.timestamp, ["x", "y", "z"]], dims=["time", "axis"], attrs=dict(units="newton/meter"))

        dataset = xr.Dataset(dict(
            position=position_array, 
            orientation=orientation_array,
            force=force_array,
            torque=torque_array,
            ), 
            attrs=dict(# male
                    file_male=self.config["task_objects"][0]["obj_name"],
                    type_male=self.config["task_objects"][0]["obj_type"],
                    material_male=self.material_male,
                    friction_male=self.friction,
                    density_male=material_list[self.material_male]["density"],
                    # female
                    file_female=self.config["task_objects"][1]["obj_name"],
                    type_female=self.config["task_objects"][1]["obj_type"],
                    material_female=self.material_female,
                    friction_female=self.friction,
                    density_female=material_list[self.material_female]["density"],))
  
        # save simulation data
        dataset.to_netcdf(path=self.savepath + ".nc")
    
        self.print_info()
    

    def print_info(self):
        print("recorded data saved to " + self.savepath)

        print("timestamp: " + str(self.timestamp.shape))
        print("eef_fts: " + str(self.eef_fts.shape))
        print("eef_pos: " + str(self.eef_pos.shape))
        print("eef_qua: " + str(self.eef_qua.shape))
        print("joint_states: " + str(self.joint_states.shape))

        print("object_type: " + self.object_type)
        print("material_male: " + self.material_male)
        print("material_female: " + self.material_female)

    def plot_data(self):
        # data = np.load(self.savepath + ".nc")
        ds = xr.load_dataset(self.savepath + ".nc")

        position = ds.position.isel(axis=[1, 2, 3]).to_numpy()
        force = ds.force.isel(axis=[1, 2, 3]).to_numpy()
        time = ds.position.time.to_numpy()

        labels = ["Fx", "Fy", "Fz", "x", "y", "z"]

        for i,j in enumerate((1, 2, 3)):
            plt.subplot(2,3,j)
            plt.plot(time, force[:,i]-force[0,i], label=labels[j-1])
            plt.legend(loc="upper right")

        for i,j in enumerate((4, 5, 6)):
            plt.subplot(2,3,j)
            plt.plot(time, position[:,i]-position[0,i], label=labels[j-1])
            plt.legend(loc="upper right")
        
        plt.tight_layout()
        plt.savefig(self.savepath + ".png")
        plt.close()
        plt.cla()
        plt.clf()

        return

   
