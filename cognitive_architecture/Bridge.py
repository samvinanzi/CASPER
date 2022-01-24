"""
Acts as a bridge between the low and high levels of the cognitive architecture.
"""

import xml.etree.ElementTree as ET
import pandas as pd
from util.PathProvider import path_provider

#BASEDIR = basedir = Path(__file__).parent.parent
#CONFIG_FILE = 'config/observation_encoding.csv'
#OBSERVATIONS_FILE = 'data/CRADLE/Observations.xml'


class ObservationData:
    def __init__(self, name, id, params):
        self.name = name
        self.id = id
        self.params = params

    def number_of_parameters(self):
        return len(self.params)


class Bridge:
    def __init__(self):
        # Loads the configuration file
        self.config = pd.read_csv(path_provider.get_encodings(), names=['name', 'id', 'param1', 'param2'])
        self.initialize_xml()

    def retrieve_data(self, observation):
        """
        Retrieves id, and list of parameters of a given observation.

        :param observation: name of the observation
        :return: ObservationData
        """
        try:
            # Locates the observation name in the configuration data
            row = self.config.loc[self.config['name'] == observation.upper()]
            name = row['name'].values[0]
            id = row['id'].values[0]
            params = []
            # Processes every parameter
            for i in range(2):
                param_name = "param" + str(i+1)
                param_i = row[param_name].values[0]
                if isinstance(param_i, str):    # If the parameter is missing, it will be a nan of type float
                    params.append(param_i)
            return ObservationData(name, id, params)
        except IndexError:
            print("Observation '{0}' not found".format(observation))
            return None

    def initialize_xml(self):
        """
        Initializes an empty observations file.

        :return: None
        """
        of = path_provider.get_observations()
        # If the file already exists, deletes the old version
        if of.is_file():
            try:
                of.unlink()
            except OSError as e:
                print("Error: %s : %s" % (of, e.strerror))
        of.touch()      # Creates the file
        # Populates the file with the root element
        root = ET.Element("Observations")
        tree = ET.ElementTree(root)
        tree.write(of, encoding="ISO-8859-1", xml_declaration=True)

    def append_observation(self, data: ObservationData, parameters):
        """
        Appends an observation to an xml file.

        :param data: ObservationData object
        :param parameters: lists of parameter names
        :return: None
        """
        assert data.number_of_parameters() == len(parameters),\
            "Needed exactly {0} parameters for observation {1}, provided {2}".\
                format(data.number_of_parameters(), data.name, len(parameters))
        tree = ET.parse(path_provider.get_observations())
        root = tree.getroot()
        # Creates one Observation sub-element
        observation = ET.SubElement(root, "Observation", attrib={'id': data.id})
        for i in range(data.number_of_parameters()):
            # Populates the parameters
            param = ET.SubElement(observation, "Param", attrib={'name': data.params[i], 'val': parameters[i]})
        tree.write(path_provider.get_observations(), encoding="ISO-8859-1", xml_declaration=True)
