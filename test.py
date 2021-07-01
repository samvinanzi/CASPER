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



#trainer = TreeTrainer()
#trainer.k_fold_cross_validation()
#trainer.train_model('all.csv', show=True)


"""
from hmmlearn.hmm import MultinomialHMM

hmm = MultinomialHMM(n_components=3, params='e')

observation_map = {
    'STILL': 0,
    'WALK': 1,
    'PICK': 2,
    'PLACE': 3,
    'TRANSPORT': 4
}

hmm.startprob_ = np.array([.33, .33, .33])
hmm.transmat_ = np.array([
    [.33, .33, .33],
    [.33, .33, .33],
    [.33, .33, .33]
])
hmm.emissionprob_ = np.array([
    [0, 0, ],
    [],
    []
])

X1 = ['PICK', 'TRANSPORT', 'PLACE']
X2 = ['PICK', 'PLACE', 'PICK', 'PLACE']
X3 = ['STILL', 'WALK', 'STILL','STILL']

X = np.concatenate([X1, X2, X3])
X = [[observation_map[x]] for x in X]
lengths = [len(X1), len(X2), len(X3)]

hmm.fit(X, lengths)

Y1 = [[observation_map[x]] for x in X1]
Y2 = [[observation_map[x]] for x in X2]
Y3 = [[observation_map[x]] for x in X3]

print("0: Pick&Place\n1: Use\n3) Relocate")

print(hmm.predict(Y1))
print(hmm.predict(Y2))
print(hmm.predict(Y3))
"""


#from random import random
#from graphviz import Digraph

#from MarkovFSM import Chain, transitions_to_graph

'''
def coin():               # random process: the perfect coin flipping
  return 1 if random() > 0.5 else 0

chain = Chain(2, coin())  # create an empty Markov chain with 2 states

for i in range(1000000):  # let the Markov chain build state transition matrix
  chain.learn(coin())   # based on 1000000 of coin flips

print(chain.get_transitions_probs(0))
'''


action1 = [
    [0, .033, .033, .9, .033],
    [.1, 0, 0, .9, 0],
    [.1, 0, 0, 0, .9],
    [.05, 0, .9, 0, .05],
    [.033, .033, 0, .9, .033]
]

action2 = [
    [0, .05, .05, .45, .45],
    [.1, 0, 0, .9, 0],
    [.1, 0, 0, 0, .9],
    [.05, 0, .05, 0, .9],
    [.05, .05, 0, .9, 0]
]

action3 = [
    [0, .9, .033, .033, .033],
    [.9, 0, 0, .1, 0],
    [.9, 0, 0, 0, .1],
    [.9, 0, .05, .0, .05],
    [.9, .05, 0, .05, 0]
]

def get_prob(model, *args):
    ret = 1
    for i, j in zip(args, args[1:]):
        ret *= model.at[i, j]
    return ret

names = ['STILL', 'WALK', 'TRANSPORT', 'PICK', 'PLACE']

chain1 = pd.DataFrame(action1, columns=names, index=names, dtype=float)
chain2 = pd.DataFrame(action2, columns=names, index=names, dtype=float)
chain3 = pd.DataFrame(action3, columns=names, index=names, dtype=float)

class Model:
    def __init__(self, chain, name):
        self.chain = chain
        self.name = name

m1 = Model(chain1, "Pick&Place")
m2 = Model(chain2, "Use")
m3 = Model(chain3, "Relocate")

def best_model(*args):
    top_score = 0.0
    top_model = None
    for model in [m1, m2, m3]:
        score = get_prob(model.chain, *args)
        if score > top_score:
            top_score = score
            top_model = model.name
    return top_model, top_score

#print(get_prob(m1.chain, 'PICK', 'TRANSPORT', 'PLACE'))
#print(get_prob(m1.chain, 'PICK', 'PLACE', 'PICK', 'PLACE'))
#print(get_prob(m1.chain, 'STILL', 'WALK', 'STILL'))

print(best_model('PICK', 'TRANSPORT', 'PLACE'))
print(best_model('PICK', 'PLACE', 'PICK', 'PLACE'))
print(best_model('STILL', 'WALK', 'STILL'))

print("\n")

print(best_model('PICK', 'PLACE', 'PICK', 'PLACE', 'PICK', 'PLACE'))


'''
transmat = [
    [0.3, 0.2, 0.5, 0.0, 0.2],
    [0.2, 0.4, 0, 0, 0.4],
    [0.5, 0.4, 0, 0.1, 0],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.6, 0.1, 0.1, 0.1, 0.1]
]

markov = pd.DataFrame(transmat, columns=['A', 'B', 'C', 'D', 'E'], index=['A', 'B', 'C', 'D', 'E'], dtype=float)
print(get_prob(markov, 'A', 'C', 'D'))
print(get_prob(markov, 'A', 'C', 'D', 'E'))
'''


print("\nDone")
pass
