import time

import cognitive_architecture.MarkovFSM
import maps.Kitchen2
import pickle
import os
import numpy as np
from cognitive_architecture import *
from util.PathProvider import path_provider
import pandas as pd
import matplotlib.pyplot as plt
from cognitive_architecture.FocusBelief import FocusBelief
from owlready2 import *
from cognitive_architecture.KnowledgeBase import Statement, KnowledgeBase


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

'''
from util.PathProvider import PathProvider

pp = PathProvider()
print(pp.get_pickle(''))
'''


def plot_focus():
    plt.gcf().set_dpi(300)
    headers = ["time", "sink", "glass", "hobs", "biscuits", "meal", "plate", "bottle"]
    file = path_provider.get_csv("focus_belief.csv")
    df = pd.read_csv(file, names=headers)
    for item in headers[1:]:
        print("Plotting {0}".format(item))
        df[["time", item]].set_index('time').plot()
        # Plot styling
        plt.legend(loc="upper right")
        plt.ylim(0, 1.0)
        plt.axhline(y=0.5, color='r', linestyle='dashed')
        # Data plotting
        plt.savefig(path_provider.get_image("{0}.png".format(item)))



"""
FOCUS SIMULATOR


focus = FocusBelief("simulator")
headers = ["time", "sink", "glass", "hobs", "biscuits", "meal", "plate", "bottle"]
file = path_provider.get_csv("test_focus.csv")
df = pd.read_csv(file, names=headers)
df = df.reset_index()
for index, row in df.iterrows():
    print("#----------- TIME {0} -----------#".format(row['time']))
    for header in headers[1:]:
        focus.raw_values[header] = row[header]  # Simulates a focus.add()
    focus.process_iteration()
    focus.print_probabilities()
    target, destination = focus.get_winners_if_exist()
    print("Target: {0}\nDestination: {1}".format(target, destination))
    #input("\nPress Enter to continue...")
"""

"""
# CRADLE-TEST

from cognitive_architecture.InternalComms import InternalComms
from cognitive_architecture.HighLevel import HighLevel
from util.PathProvider import path_provider

DOMAIN = path_provider.get_domain("Domain_kitchen_corrected.xml")
OBSERVATIONS = path_provider.get_domain("Observations_example.xml")
#OBSERVATIONS = path_provider.get_observations()

internal_comms = InternalComms()
highlevel = HighLevel(internal_comms, DOMAIN, OBSERVATIONS, debug=True)

exps = highlevel.explain()

print(exps)
"""

"""
# ONTOLOGY-TEST

onto = get_ontology('/home/samuele/Research/Ontologies/kitchen_onto.owl').load()
print("{0} PICK AND PLACE {1} {2}".format(onto.human, onto.human.pnp_target, onto.human.pnp_destination))

# This is inconsistent: a robot can't eat!
onto.tiago.eat_target = onto.meal
onto.tiago.eat_destination = onto.plate

try:
    sync_reasoner_pellet()
    print("Ontology consistent!")
except OwlReadyInconsistentOntologyError:
    print("Ontology inconsistent...")

# Removes the problem
onto.tiago.eat_target = None
onto.tiago.eat_destination = None

try:
    sync_reasoner_pellet()
    print("Ontology consistent!")
except OwlReadyInconsistentOntologyError:
    print("Ontology inconsistent...")
"""

onto = KnowledgeBase('kitchen_onto')
s = Statement("human", "COOK", "meal", "hobs")
onto.verify_statement(s, debug=True)


print("\nDone")
pass
