"""
High-Level cognitive architecture (plan recognition using a probabilistic context-free grammar)
This class incorporates and uses code from CRADLE (https://github.com/ReuthMirsky/CRADLE)
"""

import copy
import sys
import xml.etree.ElementTree as ET
import os

from CRADLE.Algorithm import ExplainAndCompute
from CRADLE.Sigma import Sigma, NT
from CRADLE.Rule import Rule
import CRADLE.PL as PL
import CRADLE.Explanation as Explanation

DATA_DIR = "data/CRADLE"


class HighLevel:
    def __init__(self, domain_file, observation_file=None, debug=False):
        # Initialization
        self.plan_library = None
        self.observations = None
        # Tries to load the domain XML file
        try:
            self.plan_library = self.readDomain(domain_file)
            if debug:
                print(self.plan_library)
        except Exception as ex:
            print("Domain File Corrupt: {0}".format(ex))
            sys.exit()
        # Tries to load the observations XML file, if provided
        if observation_file is not None:
            self.use_observation_file(observation_file)
            if debug:
                print(self.observations)

    def use_observation_file(self, observation_file):
        try:
            self.observations = self.readObservations(self.plan_library, observation_file)
        except Exception as ex:
            print("Observations File Corrupt: {0}".format(ex))
            sys.exit()

    def explain(self, debug=False):
        """
        Using the preloaded domain and observations data, tries to compute the possible explanations.

        :param debug: verbose output, True/False
        :return: list of Explanations
        """
        assert self.observations is not None, "Please run use_observation_file() before requesting an explanation"
        exps = ExplainAndCompute(self.plan_library, self.observations)
        exps.sort(key=Explanation.Explanation.getExpProbability)    # Sorts by decreasing probabilities
        if len(exps) == 0:
            print("No Explanations")
            return None
        if debug:
            n_explanations = 0
            noFrontier = 0
            while not len(exps) == 0:
                exp = exps.pop()
                print(exp)
                if exp.getFrontierSize() == 0:
                    noFrontier += 1
                n_explanations += 1
            print("Explanations: ", n_explanations)
            print("No Frontier Explanations: ", noFrontier)
        return exps

    def parse_explanations(self, exps):
        """
        Parse the list of Explanations to obtain useful information.

        :param exps: list of Explanations
        :return:
        """
        pass

    # --- Code below from CRADLE's implementation --- #

    @staticmethod
    def getLetter(listOfLetters, name):
        for letter in listOfLetters:
            if letter.get() == name:
                return letter
        return None

    def readDomain(self, domain_file):
        domain_file_path = os.path.join(DATA_DIR, domain_file)
        tree = ET.parse(domain_file_path)
        root = tree.getroot()

        # Read NTs
        ntNodes = root[0][0]
        NTs = []
        Goals = []
        for child in ntNodes:
            name = child.get('id')
            params = []
            for param in child[0]:
                params.append(param.get('name'))
            newLetter = NT(name, params)
            NTs.append(newLetter)
            if child.get('goal') == 'yes':
                Goals.append(newLetter)

        # Read Sigmas
        sigmaNodes = root[0][1]
        Sigmas = []
        for child in sigmaNodes:
            name = child.get('id')
            params = []
            for param in child[0]:
                params.append(param.get('name'))
            Sigmas.append(Sigma(name, params))

        # Read Rules
        Rules = []
        ruleNodes = root[1]

        for ruleNode in ruleNodes:
            # ruleProb = ruleNode.get('prob')
            ruleA = self.getLetter(NTs, ruleNode.get('lhs'))
            ruleOrders = []
            ruleEquals = []
            ruleRhs = []
            if ruleNode.find('Order') is not None:
                for orderConst in ruleNode.find('Order'):
                    ruleOrders.append((int(orderConst.get('firstIndex')) - 1, int(orderConst.get('secondIndex')) - 1))
            if ruleNode.find('Equals') is not None:
                for equalConst in ruleNode.find('Equals'):
                    ruleEquals.append((int(equalConst.get('firstIndex')) - 1, equalConst.get('firstParam'),
                                       int(equalConst.get('secondIndex')) - 1, equalConst.get('secondParam')))
            for child in ruleNode.findall('Letter'):
                letter = self.getLetter(NTs, child.get('id'))
                if letter is None:
                    letter = self.getLetter(Sigmas, child.get('id'))
                ruleRhs.insert(int(child.get('index')) - 1, letter)
            Rules.append(Rule(ruleA, ruleRhs, ruleOrders, ruleEquals))
        myPL = PL.PL(Sigmas, NTs, Goals, Rules)
        return myPL

    def readObservations(self, pl, observation_file):
        observations_file_path = os.path.join(DATA_DIR, observation_file)
        tree = ET.parse(observations_file_path)
        root = tree.getroot()
        observations = []
        for observation in root:
            letter = self.getLetter(pl._Sigma, observation.get('id'))
            for param in observation:
                letter.setParam(param.get('name'), param.get('val'))
            observations.append(copy.deepcopy(letter))
        return observations
