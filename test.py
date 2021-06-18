import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
from matplotlib import pyplot as plt

from EpisodeFactory import DataFrameFactory
from FocusBelief import FocusBelief

"""
factory = DataFrameFactory()
factory.reload_data()
focus = FocusBelief('human')

for i in range(40):
    new_episode = factory.build_episode(i)
    print(new_episode)
    if new_episode is None:
        continue
    else:
        objects = new_episode.get_objects_for('human')
        for object in objects:
            focus.add(object)
        if focus.has_confident_prediction():
            prediction = focus.get_top_n_items(1)
        else:
            prediction = None
        print("Prediction: *{0}* (ground truth: {1})".format(prediction, new_episode.humans['human'].target))
        focus.print_probabilities()
        print("Feature: {0}".format(new_episode.to_feature('human')))

dataset = factory.build_dataset(save=True)
"""
"""
dtypes = {
    'TIME': int,
    'MOS': bool,
    'HOLD': bool,
    'QDC': 'category',
    'QTC': 'category',
    'ACTION': str
}

#df = pd.read_csv('data\csv\dataset_clean.csv')
df = pd.read_csv('data\csv\dataset_clean.csv', dtype=dtypes)
#print(df.to_string())

qdc_mapping = {
    'TOUCH': 1,
    'NEAR': 2,
    'MEDIUM': 3,
    'FAR': 4
}
inverse_qdc_mapping = {v: k for k, v in qdc_mapping.items()}

qtc_mapping = {
    '0': 1,
    '-': 2,
    '+': 3
}
inverse_qtc_mapping = {v: k for k, v in qtc_mapping.items()}

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['ACTION']))}
inverse_class_mapping = {v: k for k, v in class_mapping.items()}

df['QDC'] = df['QDC'].map(qdc_mapping)
df['QTC'] = df['QTC'].map(qtc_mapping)

le = LabelEncoder()
y = le.fit_transform(df['ACTION'].values)

df.drop('TIME', axis=1, inplace=True)
df.drop('ACTION', axis=1, inplace=True)

print(df.to_string())
print(y)
"""

factory = DataFrameFactory()
X_train, y_train = factory.load_training_dataset('dataset_clean.csv')

clf = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
clf.fit(X_train, y_train)
tree.plot_tree(clf, feature_names=['MOS', 'HOLD', 'QDC', 'QTC'], filled=True,
               class_names=['PICK', 'PLACE', 'STILL', 'TRANSPORT', 'WALK'])
plt.show()

pass
