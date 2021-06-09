import os
import pickle

from Dataframe import *

#BASEDIR = "..\..\..\..\THRIVE++"
PICKLE_DIR = "data\pickle"


def load_pickle(filename):
    file = os.path.join(PICKLE_DIR, filename)
    data = pickle.load(open(file, "rb"))
    return data


def process(current_timestamp=1, debug=True):
    # TODO IMPORTANT! This assumes the existence of only 1 human, has to be modified in the future
    qsr_response = load_pickle('qsr_response.p')
    world_trace = load_pickle('world_trace.p')
    # Initializes an episode for the current timestamp
    ep = Episode(time=current_timestamp)
    try:
        trace_at_timestep = qsr_response.qsrs.trace[current_timestamp]
    except KeyError:    # No world trace at that timestep
        if debug:
            print("No world trace at timestamp {0}.".format(current_timestamp))
        return None
    try:
        human_trace = trace_at_timestep.qsrs['human']
        # Initialize a human frame for the current timestamp
        hf = HumanFrame()
        hf.MOS = human_trace.qsr['mos']
        hf.HOLD = world_trace.trace[current_timestamp].objects['human'].kwargs['hold']
        hf.fallback_label = world_trace.trace[current_timestamp].objects['human'].kwargs['label']
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
                of = ObjectFrame(qdc, qtc)
                hf.objects[object_name] = of
        ep.humans['human'] = hf
        pass
    except KeyError:    # No humans at that timestep
        human_trace = None
        if debug:
            print("No human data found at timestep {0}.".format(current_timestamp))
        return None
    return ep

def print_labels(episodes):
    for episode in episodes:
        try:
            print("{0}: {1}".format(episode.time, episode.humans['human'].fallback_label))
        except AttributeError:
            pass


#process(23)
episodes = []
for i in range(35):
    new_episode = process(i)
    episodes.append(new_episode)
print_labels(episodes)
pass