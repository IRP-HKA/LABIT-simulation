from src.objects.base_object import BaseObject
from src.utils.mesh_processing import MeshObjects


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
        element_contact.set("solref", " ".join([str(c) for c in self._materials[config["material"]].solref])) # "0.01 1"

        element_edge = ET.SubElement(element_flexcomp, "edge")
        element_edge.set("damping", "0.5")
        element_edge.set("equality", "true")
        # element_edge.set("solimp", "0.95 0.99 0.001 0.5 2") # 0.0001
        # element_edge.set("solref", " ".join([str(c) for c in self._materials[config["material"]]["solref"]])) # "0.01 1"

        element_plugin = ET.SubElement(element_flexcomp, "plugin")
        element_plugin.set("plugin", "mujoco.elasticity.solid")

        element_config_0 = ET.SubElement(element_plugin, "config")
        element_config_0.set("key", "young")   
        element_config_0.set("value", str(self._materials[config["material"]].young))

        element_config_1 = ET.SubElement(element_plugin, "config")
        element_config_1.set("key", "poisson")   
        element_config_1.set("value", str(self._materials[config["material"]].poisson))

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

