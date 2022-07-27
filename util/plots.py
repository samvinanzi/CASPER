import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from util.PathProvider import path_provider
from cognitive_architecture.MarkovFSM import ensemble
import time
from cognitive_architecture.PlanLibrary import GoalNotRecognizedException


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


def test_fsm(name, save=False):
    assert name.upper() in ['PNP', 'USE', 'REL'], "Name should be one of: pnp, use, rel."
    if name.upper() == 'PNP':
        observations = ['STILL', 'WALK', 'PICK', 'TRANSPORT', 'PLACE', 'PICK', 'STILL', 'WALK', 'STILL']
    elif name.upper() == 'USE':
        observations = ['STILL', 'WALK', 'PICK', 'PLACE', 'PICK', 'PLACE', 'PICK', 'PLACE', 'STILL']
    else:
        observations = ['STILL', 'WALK', 'STILL', 'WALK', 'STILL', 'WALK', 'STILL', 'WALK', 'STILL']
    filename = 'fsm_{0}.png'.format(name.lower())
    # NOISY PNP
    #observations = ['STILL', 'WALK', 'STILL', 'PICK', 'TRANSPORT', 'PLACE', 'PICK', 'STILL', 'WALK']
    m1 = []
    m2 = []
    m3 = []
    percent = None
    moment = None
    for i, obs in enumerate(observations):
        print("Inserting: {0}".format(obs))
        ensemble.add_observation(obs)
        scores = ensemble.get_scores()
        action, score, winner = ensemble.best_model()
        #print(action, score, winner)
        print(scores)
        m1.append(scores[0])
        m2.append(scores[1])
        m3.append(scores[2])
        if winner and moment is None:
            percent = round((i + 1) / len(observations), 2)
            print(percent)
            moment = i
            print("Guessed in {0}".format(i+1))
    # Plot
    x = np.array(list(range(len(observations))))
    X_ = np.linspace(x.min(), x.max(), 500)
    # plot lines
    spline_m1 = make_interp_spline(x, m1)
    spline_m2 = make_interp_spline(x, m2)
    spline_m3 = make_interp_spline(x, m3)
    M1_ = spline_m1(X_)
    M2_ = spline_m2(X_)
    M3_ = spline_m3(X_)
    plt.plot(X_, M1_, label="Pick&Place", alpha=0.7, c='blue')
    plt.plot(X_, M2_, label="Use", alpha=0.7, c='black')
    plt.plot(X_, M3_, label="Relocate", alpha=0.7, c='orange')
    #plt.plot(x, m1, label="Pick&Place", alpha=0.5, c='blue')
    #plt.plot(x, m2, label="Use", alpha=0.5, c='black')
    #plt.plot(x, m3, label="Relocate", alpha=0.5, c='red')
    # Winning moment, if exists
    if moment is not None:
        plt.axvline(x=moment, color='gray', linestyle='--', alpha=0.4)
    plt.legend(loc='lower left')
    plt.xlabel('Timestep')
    plt.ylabel('Similarity index')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)
    if not save:
        plt.show()
    else:
        plt.savefig(path_provider.get_image(filename))


def test_pl(pl, action, params):
    pl.add_observation(action, params)
    # Explanations
    tic = time.time()
    try:
        exps = pl.get_explanations()
    except GoalNotRecognizedException:
        exps = []
    toc = time.time() - tic
    ms = round(toc * 1000000, 2)
    n = len(exps)
    # Outcome
    if n == 0:
        # No explanation found
        out = None
        print("FAILURE")
    elif n == 1:
        # One explanation found
        out = exps[0]
    elif exps[0].score != exps[1].score:
        # Multiple explanations, but with a clear winner
        out = exps[0]
    else:
        # Multiple explanations with some ties between the top scored
        out = None
    # Confidence
    if n > 0:
        confidence = exps[0].score
    else:
        confidence = 0.0
    confidence = round(confidence, 2)
    print("Action: {0}\nExplanations: {1}\nOutcome: {2}\nTime: {3}\nConfidence: {4}".format(action, n, out, ms,
                                                                                            confidence))
    print("------------------")
