"""
This class is responsible for the datasets management and the construction of the Decision Tree classifier.
"""

import os

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from cognitive_architecture.EpisodeFactory import EpisodeFactory


CSV_DIR = "../data/csv"


class TreeTrainer:
    def __init__(self):
        self.factory = EpisodeFactory()
        # dtypes will be used when importing CSV in Pandas
        self.dtypes = {
            'TIME': int,
            'MOS': bool,
            'HOLD': bool,
            'QDC': 'category',
            'QTC': 'category',
            'ACTION': str
        }

    def prepare_datasets(self, min=0, max=9):
        """
        Loads the pickles on world_trace and qsr_response to build a CSV datasets and cleans it for training purposes.

        :return: None
        """
        for i in range(min, max):
            print("Operation in progress: {0}".format(i))
            factory = EpisodeFactory()
            factory.reload_data(id=i)
            for j in range(41):
                factory.build_episode(j)
            factory.build_dataset(save=True, id=i)
            print("Built...")
            factory.clean_dataset(id=i)
            print("Cleaned...")

    def combine_datasets(self, min=0, max=9):
        """
        Creates an all.csv dataset file combining all the datasetX_clean.csv files.

        :return: None
        """
        filenames = []
        for i in range(min, max):
            filenames.append(os.path.join(CSV_DIR, "dataset{0}_clean.csv".format(i)))
        # combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
        # export to csv
        combined_csv.to_csv(os.path.join(CSV_DIR, "all.csv"), index=False, encoding='utf-8-sig')

    def create_k_folds(self, min=0, max=9, debug=False):
        """
        Creates the 1-fold datasets.

        :return: None
        """
        for k in range(10):     # K will be the dataset to exclude
            filenames = []
            ids = [x for x in range(min, max) if x != k]
            for i in ids:
                filenames.append(os.path.join(CSV_DIR, "dataset{0}_clean.csv".format(i)))
            # combine all files in the list
            combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
            # export to csv
            combined_csv.to_csv(os.path.join(CSV_DIR, "kfold_exclude{0}.csv".format(k)), index=False,
                                encoding='utf-8-sig')
            if debug:
                print("K = {0}, files = {1}".format(k, filenames))

    def train_model(self, trainingset, show=False):
        """
        Trains the Decision Tree classifier using a specified dataset.

        :param trainingset: CSV dataset filename (path is implicit)
        :param show: if True, it will display the tree
        :return: the classifier
        """
        X_train, y_train = self.factory.load_training_dataset(trainingset)
        clf = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
        clf.fit(X_train, y_train)
        clf.classes_ = np.array(['PICK', 'PLACE', 'STILL', 'TRANSPORT', 'WALK'])
        if show:
            plt.figure(figsize=(12, 12))
            tree.plot_tree(clf, feature_names=['MOS', 'HOLD', 'QDC', 'QTC'], filled=True,
                           class_names=['PICK', 'PLACE', 'STILL', 'TRANSPORT', 'WALK'], fontsize=11)
            plt.savefig('..\\..\\images\\tree.png', dpi=100)
            plt.show()
        return clf

    def k_fold_cross_validation(self, min=0, max=9, debug=False):
        """
        Performs 1-fold cross-validation.

        :return: None
        """
        scores = []
        for k in range(10):
            train_dataset = 'kfold_exclude{0}.csv'.format(k)
            test_dataset = 'dataset{0}_clean.csv'.format(k)
            clf = self.train_model(train_dataset)
            X_test, y_test = self.factory.load_training_dataset(test_dataset)
            score = clf.score(X_test, y_test)
            print("K = {0}, score = {1}".format(k, score))
            scores.append(score)
        scores = np.array(scores)
        avg_score = scores.mean()
        if debug:
            print("1-fold cross validation. Average score: {0}".format(avg_score))
        return avg_score
