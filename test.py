import time

from cognitive_architecture.FocusBelief import FocusBelief
from cognitive_architecture.TreeTrainer import TreeTrainer
import pickle
import os
import numpy as np
from cognitive_architecture.Episode import Episode
from cognitive_architecture.EpisodeFactory import EpisodeFactory
from cognitive_architecture.MarkovFSM import ensemble
from cognitive_architecture.HighLevel import HighLevel
from cognitive_architecture.Bridge import Bridge
from cognitive_architecture.InternalComms import InternalComms

'''
factory = DataFrameFactory()
X_train, y_train = factory.load_training_dataset('dataset_clean.csv')

clf = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
clf.fit(X_train, y_train)
tree.plot_tree(clf, feature_names=['MOS', 'HOLD', 'QDC', 'QTC'], filled=True,
               class_names=['PICK', 'PLACE', 'STILL', 'TRANSPORT', 'WALK'])
plt.show()
'''

'''
trainer = TreeTrainer()
#trainer.k_fold_cross_validation()
tree = trainer.train_model('all.csv', show=False)
pickle.dump(tree, open("data\\pickle\\tree.p", "wb"))
'''

'''
tree = pickle.load(open("data\\pickle\\tree.p", "rb"))
factory = EpisodeFactory()
factory.reload_data(0)
for i in range(0, 40):
    episode = factory.build_episode(i)
    if episode is not None:
        feature = episode.to_feature(human="human", train=False)
        prediction = tree.predict(feature)[0]
        ensemble.add_observation(prediction)
        action, score, winner = ensemble.best_model()
        print("{0}) {1} = {2}".format(i, feature, prediction))
        if winner:
            print("\tAction: {0}".format(action))
'''

'''
from cognitive_architecture.EpisodeFactory import EpisodeFactory

focus = FocusBelief(human_name="human")

factory = EpisodeFactory()
factory.reload_data(0)
ep = factory.build_episode(15)
objects = ep.get_objects_for("human")
for object in objects:
    focus.add(object)
if focus.has_confident_prediction():
    target = focus.get_top_n_items(1)
    target_name, target_score = list(target.items())[0]
print("Focus: {0}".format(target))
ep.humans["human"].target = target_name
feature = ep.to_feature(human="human", train=False)
print(feature)


context = list(focus.get_top_n_items(2))[1]
'''


'''
bridge = Bridge()
data = bridge.retrieve_data("pick and place")
bridge.append_observation(data, ['biscuits', 'plate'])
data = bridge.retrieve_data("eat")
bridge.append_observation(data, ['biscuits'])

ic = InternalComms()

hl = HighLevel(ic, 'Domain_kitchen.xml', debug=True)
hl.use_observation_file('Observations.xml')
exps = hl.explain(debug=False)
hl.parse_explanations(exps)
'''

'''
ic = InternalComms()
hl = HighLevel(ic, 'Domain_kitchen.xml')
hl.start()
bridge = Bridge()
time.sleep(1)
print("About to insert pnp...")
data = bridge.retrieve_data("pick and place")
bridge.append_observation(data, ['biscuits', 'plate'])
ic.put(True)
time.sleep(2)
print("About to insert eat...")
data = bridge.retrieve_data("eat")
bridge.append_observation(data, ['biscuits'])
ic.put(True)
'''

from util.PathProvider import PathProvider

pp = PathProvider()
print(pp.get_pickle(''))

print("\nDone")
pass
