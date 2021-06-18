"""
This class handles the construction of Episodes
"""

import os
import pickle
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Episode import *

PICKLE_DIR = "data\pickle"
CSV_DIR = "data\csv"


class DataFrameFactory:
    def __init__(self, world_trace=None, qsr_response=None):
        self.world_trace = world_trace
        self.qsr_response = qsr_response
        self.episodes = []

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

    def reload_data(self):
        """
        Realoads previously saved data.

        :return: None
        """
        self.world_trace = self.load_pickle('world_trace.p')
        self.qsr_response = self.load_pickle('qsr_response.p')

    def build_episode(self, current_timestamp=1, debug=True):
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
                    # Create an object frame for this human-object couple
                    of = ObjectFrame(object_name, qdc, qtc)
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

    def build_dataset(self, save=False):
        """
        Builds a training/testing dataset for ML applications.

        :param save: optional parameter, if true it will save the data in both pickle and CSV
        :return: A 5xM matrix, where M is the number of recorded episodes.
        """
        dataset = []
        for episode in self.episodes:
            dataset.append(episode.to_feature('human'))
        if save:
            # Pickle
            pickle.dump(dataset, open(os.path.join(PICKLE_DIR, "dataset.p"), "wb"))

            # CSV
            with open(os.path.join(CSV_DIR, "dataset.csv"), 'w', encoding='UTF8', newline='') as f:
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
            dtypes = {
                'TIME': int,
                'MOS': bool,
                'HOLD': bool,
                'QDC': 'category',
                'QTC': 'category',
                'ACTION': str
            }
            df = pd.read_csv(file, dtype=dtypes)

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
