import multiprocessing.connection
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from cognitive_architecture.KnowledgeBase import KnowledgeBase, ObservationStatement
from util.PathProvider import path_provider
import pickle
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from cognitive_architecture.MarkovFSM import ensemble
import util.plots


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
kb = KnowledgeBase('kitchen_onto')
os1 = ObservationStatement("human", "COOK", "meal", "hobs")
os2 = ObservationStatement("tiago", "EAT", "meal", "plate")
os3 = ObservationStatement("human", "COOK", "meal", "sink")
kb.verify_observation(os1, debug=True)
kb.verify_observation(os2, debug=True)
kb.verify_observation(os3, debug=True)
"""

"""
# ONTOLOGY-TEST

onto = KnowledgeBase('kitchen_onto')
ob_s = ObservationStatement("human", "SIP", "glass", "bottle")
onto.verify_observation(ob_s, debug=True, infer=True)
#g_s = GoalStatement("human", "LUNCH", "biscuits")
#onto.verify_goal(g_s, debug=True)
"""

"""
# PLAN-LIBRARY TEST

pl = PlanLibrary()
pl.add_observation("Pick&Place", parameters={'item': 'meal', 'destination': 'hobs'})
#pl.add_observation("Pick&Place", parameters={'item': 'meal', 'destination': 'plate'})
#pl.add_observation("Cook", parameters={'food': 'meal', 'appliance': 'hobs'})
#pl.add_observation("Sip", parameters={'beverage': 'water', 'vessel': 'glass'})
#pl.add_observation("Pick&Place", parameters={'item': 'biscuits', 'destination': 'plate'})
#pl.add_observation("Eat", parameters={'food': 'biscuits', 'vessel': 'plate'})

explanations = pl.get_explanations(render=True)
#print(explanations[2].get_frontier())

#for pl in pl.plans:
#    pl.dot_render()
"""

'''
hl = HighLevel()
goal: Plan = hl.process(observation="Pick&Place", parameters={'target': 'meal', 'destination': 'hobs'})
#goal: Plan = hl.process(observation="Pick&Place", parameters={'item': 'meal', 'destination': 'hobs'})
#goal: Plan = hl.process(observation="Pick&Place", parameters={'item': 'meal', 'destination': 'hobs'})
goal.render()
'''

# ACTION COLLABORATION

#goal = pickle.load(open(path_provider.get_save('GOALTREE.p'), "rb"))

#plan = make_plan(goal)

#util.plots.test_fsm('pnp', save=True)
#util.plots.test_fsm('use', save=True)
#util.plots.test_fsm('rel', save=True)


"""
from cognitive_architecture.PlanLibrary import PlanLibrary

pl = PlanLibrary()


action = "WASH"
params = {
    'target': 'meal',
    'destination': 'hobs'
}
util.plots.test_pl(pl, action, params)

action = "COOK"
params = {
    'target': 'meal',
    'destination': 'plate'
}
util.plots.test_pl(pl, action, params)
"""

from util.Logger import Logger

logger = Logger()
data = {
    'trial': 2,
    'observed_actions': 3,
    'time': 50,
    'prediction': 'breakfast',
    'collab_len': 2
}
logger.log(data)
logger.log(data)
logger.log(data)
logger.log(data)
logger.log(data)    #4


print("\nDone")
pass
