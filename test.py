import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
from matplotlib import pyplot as plt

from EpisodeFactory import EpisodeFactory
from FocusBelief import FocusBelief
from TreeTrainer import TreeTrainer

"""
# Preparing all datasets
for i in range(10):
    print("Operation in progress: {0}".format(i))
    factory = DataFrameFactory()
    factory.reload_data(id=i)
    for j in range(41):
        factory.build_episode(j)
    factory.build_dataset(save=True, id=i)
    print("Built...")
    factory.clean_dataset(id=i)
    print("Cleaned...")
"""

'''
# Combining the datasets
CSV_DIR = "data\csv"
filenames = []
for i in range(10):
    filenames.append(os.path.join(CSV_DIR, "dataset{0}_clean.csv".format(i)))
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
#export to csv
combined_csv.to_csv(os.path.join(CSV_DIR, "all.csv"), index=False, encoding='utf-8-sig')
'''

'''
# Creates the K-folds
CSV_DIR = "data\csv"
for k in range(10):
    filenames = []
    ids = [x for x in range(10) if x != k]
    for i in ids:
        filenames.append(os.path.join(CSV_DIR, "dataset{0}_clean.csv".format(i)))
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
    # export to csv
    combined_csv.to_csv(os.path.join(CSV_DIR, "kfold_exclude{0}.csv".format(k)), index=False, encoding='utf-8-sig')
    print("K = {0}, files = {1}".format(k, filenames))
'''

'''
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
'''
'''
factory = DataFrameFactory()
X_train, y_train = factory.load_training_dataset('dataset_clean.csv')

clf = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
clf.fit(X_train, y_train)
tree.plot_tree(clf, feature_names=['MOS', 'HOLD', 'QDC', 'QTC'], filled=True,
               class_names=['PICK', 'PLACE', 'STILL', 'TRANSPORT', 'WALK'])
plt.show()
'''



trainer = TreeTrainer()
#trainer.k_fold_cross_validation()
trainer.train_model('all.csv', show=True)

pass
