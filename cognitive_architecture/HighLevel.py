"""
High-Level cognitive architecture (plan recognition using a probabilistic context-free grammar)
This class incorporates and uses code from CRADLE (https://github.com/ReuthMirsky/CRADLE)
"""

import copy
import sys
import xml.etree.ElementTree as ET
from cognitive_architecture.StopThread import StopThread
from CRADLE.Algorithm import ExplainAndCompute
from CRADLE.Sigma import Sigma, NT
from CRADLE.Rule import Rule
import CRADLE.PL as PL
import CRADLE.Explanation as Explanation
from cognitive_architecture.InternalComms import InternalComms
from cognitive_architecture.KnowledgeBase import kb, GoalStatement, ObservationStatement
from util.PathProvider import path_provider

DATA_DIR = "data/CRADLE"
OBSERVATIONS_FILE = "Observations.xml"


class Goal:
    def __init__(self, name=None, probability=0.0, param=None, frontier=None):
        self.name = name
        self.probability = probability
        self.param = param
        self.frontier = frontier
        self.exp = None     # Saves the explanation from which it derived

    def parse_from_explanation(self, exp: Explanation):
        self.exp = exp
        goal_node = exp.getTrees()[0].getRoot()
        self.name = goal_node._ch
        self.param = goal_node._params
        self.frontier = []
        frontier = exp.getTrees()[0].getFrontier()
        for element in frontier:
            self.frontier.append(ObservationStatement.from_NT(element.getRoot()))
        self.probability = exp.getExpProbability()

    def to_goal_statement(self):
        # todo this assumes the existence of only one human, has to be changed in the future
        return GoalStatement("human", self.name, self.param[0][1], self.frontier)

    def validate(self):
        return kb.verify_goal(self.to_goal_statement())

    def __str__(self):
        return "{0}\nProbability: {1}".format(str(self.to_goal_statement()), self.probability)


class HighLevel(StopThread):
    def __init__(self, internal_comms, domain_file, observation_file=None, debug=False):
        StopThread.__init__(self)
        # Initialization
        self.plan_library = None
        self.observations = None
        self.internal_comms: InternalComms = internal_comms
        self.debug = debug
        # Tries to load the domain XML file
        try:
            self.plan_library = self.readDomain(domain_file)
            if self.debug:
                print(self.plan_library)
        except Exception as ex:
            print("Domain File Corrupt: {0}".format(ex))
            sys.exit()
        # Tries to load the observations XML file, if provided
        if observation_file is not None:
            self.use_observation_file(observation_file)
            if self.debug:
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
        #exps.sort(key=Explanation.Explanation.getExpProbability)    # Sorts by decreasing probabilities
        if len(exps) == 0:
            print("No Explanations")
            return None
        if debug:
            n_explanations = 0
            noFrontier = 0
            for exp in exps:
                print(exp)
                if exp.getFrontierSize() == 0:
                    noFrontier += 1
                n_explanations += 1
            print("Explanations: ", n_explanations)
            print("No Frontier Explanations: ", noFrontier)
        return exps

    def parse_explanations(self, exps, debug=False):
        """
        Parses the explanations produced by CRADLE and extracts information on the uderlying goals.

        @param exps: list of Explanations
        @param debug: if True, activates verbose output
        @return: one Goal, or None
        """
        goals = []
        for exp in exps:
            new_goal = Goal()
            new_goal.parse_from_explanation(exp)
            if debug:
                print("Goal: {0}\nValid: {1}\n".format(new_goal, new_goal.validate()))
            gs = new_goal.to_goal_statement()
            if kb.verify_goal(gs):
                kb.infer_frontier(new_goal.to_goal_statement())
                goals.append(new_goal)
            elif debug:
                print("Filtered out goal \"{0}\"".format(gs))
        n_goals = len(goals)
        if n_goals == 0:
            print("I'm sorry, I can't explain what's happening.")
            return None
        elif n_goals > 2:
            print("Too many possibilities, I'm not sure...")
            return None
        elif n_goals == 2 and goals[0].probability == goals[1].probability:
            print("Two equal probabilities, not sure yet...")
            return None
        else:   # 2 goals with different probabilities or 1 goal
            print("I have found an explanation!")
            return goals[0]

    def run(self) -> None:
        self.stop_flag = False  # This is done to avoid unexpected behavior
        if self.debug:
            print("[DEBUG] " + self.__class__.__name__ + " thread is running in background.")
        while not self.stop_flag:
            # Retrieves a new observation, when available
            observation = self.internal_comms.get()  # Blocking call
            if observation is False:
                # The goal is unknown
                return
            else:   # An observation was appended
                self.use_observation_file(path_provider.get_observations())    # Recalculate observations from the file
                exps = self.explain(debug=False)    # try to explain them
                if self.debug:
                    print("[HL] Possible explanations:\n{0}".format(exps))
                if exps is not None:
                    goal = self.parse_explanations(exps, debug=False)
                    # Report the findings back to the low-level
                    self.internal_comms.write_goal(goal)
                    if goal:
                        if self.debug:
                            print("[HL] Goal detected! {0}".format(str(goal)))
                        # At this point, it must stop executing
                        #self.stop()     # todo sure?
                else:
                    # No explanation can describe the observations
                    print("[CRITICAL] No explanation found for the observed events!")
                    #self.stop()

# **********************************************************************************************************************

    # --- Code below from CRADLE's implementation --- #

    @staticmethod
    def getLetter(listOfLetters, name):
        for letter in listOfLetters:
            if letter.get() == name:
                return letter
        return None

    def readDomain(self, domain_file):
        #domain_file_path = os.path.join(DATA_DIR, domain_file)
        #domain_file_path = Path(DATA_DIR / domain_file).resolve()
        tree = ET.parse(path_provider.get_domain(domain_file))
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
        #observations_file_path = os.path.join(DATA_DIR, observation_file)
        tree = ET.parse(path_provider.get_observations())
        root = tree.getroot()
        observations = []
        for observation in root:
            letter = self.getLetter(pl._Sigma, observation.get('id'))
            for param in observation:
                letter.setParam(param.get('name'), param.get('val'))
            observations.append(copy.deepcopy(letter))
        return observations
