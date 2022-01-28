"""
This class handles the construction of Episodes
"""

import os
import pickle
import numpy as np
import csv
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from cognitive_architecture.Episode import *

np.seterr(divide='ignore', invalid='ignore')    # Suppresses the RuntimeWarning error on a handled divide-by-zero case

PICKLE_DIR = "data\pickle"
CSV_DIR = "data\csv"


class EpisodeFactory:
    def __init__(self, world_trace=None, qsr_response=None):
        self.world_trace = world_trace
        self.qsr_response = qsr_response
        self.episodes = []
        self.dtypes = {
                'TIME': int,
                'MOS': bool,
                'HOLD': bool,
                'QDC': 'category',
                'QTC': 'category',
                'ACTION': str
            }

    @staticmethod
    def load_pickle(filename):
        """
        Loads pickled data from a specific filename in the base directory.

        :param filename: str
        :return: unpickled data
        """
        file = os.path.join(PICKLE_DIR, filename)
        if os.path.exists(file) and os.path.isfile(file):
            data = pickle.load(open(file, "rb"))
            return data
        else:
            print("Invalid filename: {0}".format(file))
            quit(-1)

    def reload_data(self, id=0):
        """
        Realoads previously saved data.

        :return: None
        """
        self.world_trace = self.load_pickle('world_trace{0}.p'.format(id))
        self.qsr_response = self.load_pickle('qsr_response{0}.p'.format(id))

    def build_episode(self, current_timestamp=1, debug=False):
        """
        This method builds an episode for the current timestamp, given the world data.

        :param current_timestamp: positive integer.
        :param debug: True/False
        :return: generated Episode
        """
        assert current_timestamp >= 0, "Timestamp parameter must be >= 0."
        if self.world_trace is None or self.qsr_response is None:
            self.reload_data()
        # TODO IMPORTANT! This assumes the existence of only 1 human, has to be modified in the future
        # Initializes an episode for the current timestamp
        ep = Episode(time=current_timestamp)
        try:
            trace_at_timestep = self.qsr_response.qsrs.trace[current_timestamp]
        except KeyError:    # No world trace at that timestep
            if debug:
                print("No world trace at timestamp {0}.".format(current_timestamp))
            return None
        try:
            human_trace = trace_at_timestep.qsrs['human']
            # Initialize a human frame for the current timestamp
            hf = HumanFrame('human')
            hf.x = self.world_trace.trace[current_timestamp].objects['human'].x
            hf.y = self.world_trace.trace[current_timestamp].objects['human'].y
            hf.ov = self.world_trace.trace[current_timestamp].objects['human'].kwargs['ov']
            hf.MOS = human_trace.qsr['mos'].upper()
            hf.HOLD = self.world_trace.trace[current_timestamp].objects['human'].kwargs['hold']
            label = self.world_trace.trace[current_timestamp].objects['human'].kwargs['label']
            target = self.world_trace.trace[current_timestamp].objects['human'].kwargs['target']
            hf.target = target
            if target is None or target == '':
                hf.fallback_label = label
            # Find the objects with which the humans has a spatial relation
            keys = trace_at_timestep.qsrs.keys()
            for key in keys:
                name_split = key.split(",")
                if len(name_split) == 2:
                    object_name = name_split[-1]
                    object_trace = trace_at_timestep.qsrs[key]
                    qdc = object_trace.qsr['argd']
                    qtc = object_trace.qsr['qtcbs']
                    # Adjustments
                    qdc = qdc.upper()
                    qtc = qtc[0]
                    x = self.world_trace.trace[current_timestamp].objects[object_name].x
                    y = self.world_trace.trace[current_timestamp].objects[object_name].y
                    # Calculate the orientation angle wrt the human
                    object_vector = np.asarray([hf.x - x, hf.y - y])
                    try:
                        unit_vector_1 = hf.ov / np.linalg.norm(hf.ov)
                        unit_vector_2 = object_vector / np.linalg.norm(object_vector)
                        dot_product = np.dot(unit_vector_1, unit_vector_2)
                        angle = round(math.degrees(np.arccos(dot_product)))
                    except (ValueError, RuntimeError):
                        angle = 0.0     # This happens when an object is held, i.e. when it shares the human's position
                    # Create an object frame for this human-object couple
                    of = ObjectFrame(object_name, qdc, qtc, x=x, y=y, theta=angle)
                    if target == object_name:
                        of.label = label
                    hf.objects[object_name] = of
            ep.humans['human'] = hf
            pass
        except KeyError:    # No humans at that timestep
            human_trace = None
            if debug:
                print("No human data found at timestep {0}.".format(current_timestamp))
            return None
        self.episodes.append(ep)
        return ep

    def angle_with(self, object, orientation_vector, human_position):
        """
        Calculates the angle between the human's orientation and a specified object.

        :param object: ObjectFrame of interest
        :param orientation_vector: The human's orientation vector, calculated previously.
        :param human_position list with the [x,y] position of the human
        :return Angle, in degrees
        """
        # Calculate the human-to-object vector
        object_position = np.asarray([object.x, object.y])
        dx = human_position[0] - object_position[0]
        dy = human_position[1] - object_position[1]
        object_vector = np.asarray([dx, dy])
        try:
            unit_vector_1 = orientation_vector / np.linalg.norm(orientation_vector)
            unit_vector_2 = object_vector / np.linalg.norm(object_vector)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)
            angle_deg = round(math.degrees(angle))
        except RuntimeError:
            angle_deg = 0.0
        return angle_deg

    def get_episode_at_timestamp(self, timestamp):
        """
        Retrieves the episode at a given timestamp.

        :param timestamp: integer
        :return: The found episode or None.
        """
        assert timestamp >= 0, "Timestamp parameter must be >= 0."
        for episode in self.episodes:
            if episode.time == timestamp:
                return episode
        return None

    def print_all_labels(self):
        """
        Prints all the episodes.

        :return: None
        """
        for episode in self.episodes:
            try:
                print(episode)
            except AttributeError:
                pass

    def build_dataset(self, save=False, id=0):
        """
        Builds a training/testing dataset for ML applications.

        :param save: optional parameter, if true it will save the data in both pickle and CSV
        :return: A 5xM matrix, where M is the number of recorded episodes.
        """
        dataset = []
        for episode in self.episodes:
            dataset.append(episode.to_train_feature('human'))
        if save:
            # Pickle
            pickle.dump(dataset, open(os.path.join(PICKLE_DIR, "dataset{0}.p".format(id)), "wb"))

            # CSV
            with open(os.path.join(CSV_DIR, "dataset{0}.csv".format(id)), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # Write the header
                writer.writerow(['TIME', 'MOS', 'HOLD', 'QDC', 'QTC', 'ACTION'])
                for row in dataset:
                    writer.writerow(row)

            print("Dataset saved (pickle/CSV) in {0}".format(PICKLE_DIR))
        return np.array(dataset)

    def load_training_dataset(self, filename):
        file = os.path.join(CSV_DIR, filename)
        if not os.path.exists(file) or not os.path.isfile(file):
            print("Error trying to access data in {0}".format(file))
        else:
            df = pd.read_csv(file, dtype=self.dtypes)

            qdc_mapping = {
                'TOUCH': 1,
                'NEAR': 2,
                'MEDIUM': 3,
                'FAR': 4,
                'IGNORE': 5
            }
            #inverse_qdc_mapping = {v: k for k, v in qdc_mapping.items()}

            qtc_mapping = {
                '0': 1,
                '-': 2,
                '+': 3,
                'IGNORE': 4
            }
            #inverse_qtc_mapping = {v: k for k, v in qtc_mapping.items()}

            #class_mapping = {label: idx for idx, label in enumerate(np.unique(df['ACTION']))}
            #inverse_class_mapping = {v: k for k, v in class_mapping.items()}

            df['QDC'] = df['QDC'].map(qdc_mapping)
            df['QTC'] = df['QTC'].map(qtc_mapping)

            le = LabelEncoder()
            y = le.fit_transform(df['ACTION'].values)
            class_names = le.inverse_transform(y)

            df.drop('TIME', axis=1, inplace=True)
            df.drop('ACTION', axis=1, inplace=True)

            return df.to_numpy(), y

    def clean_dataset(self, id=0):
        """
        Reads a csv dataset, cleans the data for training and saves it in another CSV file.

        :param id: dataset id
        :return: None
        """
        filename = os.path.join(CSV_DIR, 'dataset{0}.csv'.format(id))
        if not os.path.exists(filename) or not os.path.isfile(filename):
            print("{0} is not a valid csv filename".format(filename))
        else:
            df = pd.read_csv(filename, dtype=self.dtypes)
            for index, row in df.iterrows():
                action = row['ACTION']
                if action == 'STILL':
                    df.loc[index, 'MOS'] = False
                if action == 'WALK':
                    df.loc[index, 'MOS'] = True
                if action == 'PICK':
                    df.loc[index, 'MOS'] = False
                if action == 'PLACE':
                    df.loc[index, 'MOS'] = False
                if action == 'TRANSPORT':
                    df.loc[index, 'MOS'] = True
            df.to_csv(os.path.join(CSV_DIR, 'dataset{0}_clean.csv'.format(id)), index=False)
