"""

This class uses an ontology to verify the correctness of the low-level predictions.

"""

from owlready2 import *
from util.PathProvider import path_provider
import multiprocessing
import traceback
import os
import contextlib

# Suppression of owlready2 logs like '* Owlready2 * Pellet took 0.6774663925170898 seconds'
owlready2.JAVA_EXE = "/usr/bin/java"  # Ensure correct Java
owlready2.reasoning.log = False       # Suppress verbose Pellet logging

# Singleton holder for local threading lock
_LOCAL_LOCK_SINGLETON = None
_FASTENERS_LOCK_SINGLETON = None

# Attempt to import a robust inter-process lock. If not available, fall back to a no-op lock.
try:
    import fasteners
    _HAS_FASTENERS = True
except Exception:
    _HAS_FASTENERS = False

class ObservationStatement:
    def __init__(self, actor, action, target, destination):
        # For example: human PNP meal hobs
        self.actor = actor
        self.action = action
        self.target = target
        self.destination = destination
        self.valid = None
        # Corrects the "pick and place" action label, because it appears differently elsewhere:
        if self.action.lower() == "pick and place":
            self.action = "PNP"

    def __str__(self):
        return "{0} {1} {2} {3}".format(self.actor, self.action.upper(), self.target, self.destination)

    def __eq__(self, other):
        assert isinstance(other, ObservationStatement)
        return True if self.actor == other.actor and self.action == other.action and self.target == other.target and \
                       self.destination == other.destination else False


class GoalStatement:
    def __init__(self, actor, goal, target, frontier=None):
        assert actor is not None and goal is not None and target is not None, \
            "All the fields are mandatory and cannot be None."
        # For example: human LUNCH meal
        self.actor = actor
        self.goal = goal
        self.target = target
        self.frontier = frontier
        self.valid = None

    def __str__(self):
        return "{0} {1} {2}\nFrontier: {3}".format(self.actor, self.goal.upper(), self.target,
                                                   [str(f) for f in self.frontier])

    def __eq__(self, other):
        assert isinstance(other, GoalStatement)
        return True if self.goal == other.goal and self.target == other.target else False


def clear_sqlite_locks(db_path):
    for suffix in ["-shm", "-wal", "-journal"]:
        f = str(db_path) + suffix
        if os.path.exists(f):
            print(f"[CLEANUP] Removing stale lock file: {f}")
            os.remove(f)
            
class KnowledgeBase:
    # Saves the world in an SQLite3 file on disk instead of keeping it in memory
    db = path_provider.get_SQLite3()
    #clear_sqlite_locks(db)
    default_world.set_backend(filename=db, exclusive=True)

    def __init__(self, ontology_name):
        ontofile = path_provider.get_ontology(ontology_name)
        self.onto = get_ontology(ontofile).load()
        default_world.save()
        self.op = [property.name for property in self.onto.object_properties()]     # All the object properties names
        self.individuals = [individual for individual in self.onto.individuals()]   # All the individuals
        self.history = {'valid': [], 'invalid': []}
   
    def find_individual(self, name):
        """
        Finds a specific individual by name.

        @param name: name to search for
        @return: the individual, or None if not found
        """
        result = [i for i in self.individuals if i.name == name]
        if result:
            return result[0]
        else:
            return None

    def check_history(self, statement):
        """
        To save resources, checks if this specific statement hasn't already been processed in the past.

        @param statement: ObservationStatement or GoalStatement
        @return: True or False if found, None if not yet processed
        """
        assert isinstance(statement, ObservationStatement)
        if statement in self.history['valid']:
            return True
        elif statement in self.history['invalid']:
            return False
        else:
            return None


    def infer_frontier(self, goal):
        """
        This function allows the reasoner to fill in some details that might be missing from the CRADLE explanation.

        @param goal: GoalStatement which contains the frontier to be analyzed
        @return: None
        """
        assert isinstance(goal, GoalStatement), "goal has to be an instance of GoalStatement."
        for f in goal.frontier:
            self.verify_observation(f, infer=True)
    
    def _get_ipc_lock(self):
        """
        Return an InterProcessLock context manager. Use fasteners if available.
        Singleton-style: returns the same lock every time.
        """
        global _LOCAL_LOCK_SINGLETON, _FASTENERS_LOCK_SINGLETON

        lock_path = os.environ.get("ONTOLOGY_LOCK_PATH", "/tmp/casper_ontology.lock")

        if _HAS_FASTENERS:
            if _FASTENERS_LOCK_SINGLETON is None:
                import fasteners
                _FASTENERS_LOCK_SINGLETON = fasteners.InterProcessLock(lock_path)
            return _FASTENERS_LOCK_SINGLETON
        else:
            # Fallback to a shared threading lock
            if _LOCAL_LOCK_SINGLETON is None:
                _LOCAL_LOCK_SINGLETON = threading.Lock()

            @contextlib.contextmanager
            def _dummy_lock():
                _LOCAL_LOCK_SINGLETON.acquire()
                try:
                    yield
                finally:
                    _LOCAL_LOCK_SINGLETON.release()

            return _dummy_lock()

    def _refresh_individual_cache(self):
        """
        Refresh cached individuals and object property names.
        Must be called whenever ontology might have changed externally.
        This method is now thread/process safe.
        """
        lock = self._get_ipc_lock()
        with lock:
            # Refresh cache
            self.individuals = [ind for ind in self.onto.individuals()]
            self.op = [p.name for p in self.onto.object_properties()]

    def _is_known_individual(self, obj):
        """
        Return True if the object is one of the ontology individuals.
        Accesses the cache safely under IPC lock.
        """
        if obj is None:
            return False

        lock = self._get_ipc_lock()
        with lock:
            # Ensure cache exists
            if not hasattr(self, "individuals") or self.individuals is None:
                self._refresh_individual_cache()

            # Compare by identity (preferred) or by name fallback
            for ind in self.individuals:
                if obj is ind or getattr(obj, "name", None) == getattr(ind, "name", None):
                    return True

        return False


    # ---------------------------
    # _verify: run the reasoner IN THIS PROCESS
    # ---------------------------
    def _verify(self, engine="pellet", infer=False, debug=False):
        """
        Run the reasoner inside the current process (no multiprocess spawn).
        Returns True if consistent, False otherwise.
        """
        if debug:
            print(f"[KNOWLEDGE BASE] Running reasoner ({engine}), infer={infer}, debug={debug}")
        try:
            # Save world for debugging if possible
            try:
                default_world.save()
            except Exception:
                if debug:
                    print("Warning: default_world.save() failed (continuing).")
            with open(os.devnull, "w") as fnull:# Redirect Owlread and Java messages
                with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                    if engine == "hermit":
                        sync_reasoner_hermit(self.onto, infer_property_values=infer, debug=(1 if debug else 0))
                    else:
                        sync_reasoner_pellet(self.onto, infer_property_values=infer, debug=(1 if debug else 0))
            return True
        except OwlReadyInconsistentOntologyError:
            if debug:
                print("Ontology inconsistent detected by reasoner.")
            return False
        except Exception as e:
            if debug:
                print("Reasoner raised exception:", repr(e))
                traceback.print_exc()
            return False

    # ---------------------------
    # verify_observation (adapted, defensive)
    # ---------------------------
    def verify_observation(self, observation_statement, infer=False, debug=False):
        """
        Verify if an observation statement is consistent with the ontology.
        Defensive: only assigns Individuals; runs reasoner under inter-process lock.

        @param observation_statement: An ObservationStatement object representing the action of an actor in the world.
        @param infer: if True, enables inference on data properties.
        @param debug: Verbose output
        @return: True or False
        """
        assert isinstance(observation_statement, ObservationStatement), "Input must be an ObservationStatement."

        # History short-circuit
        if not infer:
            observation_statement.valid = self.check_history(observation_statement)
            if observation_statement.valid is not None:
                return observation_statement.valid

        # Find ontology individuals
        actor = self.find_individual(observation_statement.actor)
        action = observation_statement.action
        target = self.find_individual(observation_statement.target)
        destination = self.find_individual(observation_statement.destination)

        if debug:
            print("[KNOWLEDGE BASE] verify_observation:", actor, action, target, destination)

        # Compose property names
        p = action.lower()
        pt = p + "_target"
        pd = p + "_destination"

        if pt not in self.op or pd not in self.op:
            if debug:
                print("Action not found:", action, "expected properties:", pt, pd)
            observation_statement.valid = False
            return False

        if actor is None:
            if debug:
                print("Actor not found in ontology:", observation_statement.actor)
            observation_statement.valid = False
            return False

        if target is not None and not self._is_known_individual(target):
            if debug:
                print("Target is not a known individual:", observation_statement.target)
            observation_statement.valid = False
            return False

        if destination is not None and not self._is_known_individual(destination):
            if debug:
                print("Destination is not a known individual:", observation_statement.destination)
            observation_statement.valid = False
            return False

        # Run under IPC lock
        lock = self._get_ipc_lock()
        with lock:
            try:
                # Temporary assignments
                setattr(actor, pt, target)
                if destination is not None:
                    setattr(actor, pd, destination)

                # Run reasoner
                observation_statement.valid = self._verify(engine="pellet", infer=infer, debug=debug)

                # Refresh names if inference enabled
                if infer:
                    observation_statement.target = getattr(actor, pt).name if getattr(actor, pt) else None
                    observation_statement.destination = getattr(actor, pd).name if getattr(actor, pd) else None

                # Update history
                if observation_statement.valid:
                    self.history.setdefault('valid', []).append(observation_statement)
                else:
                    self.history.setdefault('invalid', []).append(observation_statement)

            finally:
                # Reset temporary properties
                try:
                    setattr(actor, pt, None)
                    setattr(actor, pd, None)
                except Exception:
                    if debug:
                        print("Warning: failed to reset temporary properties.")

        if debug:
            print("[KNOWLEDGE BASE] Observation consistent?", observation_statement.valid)
        return observation_statement.valid

    # ---------------------------
    # verify_goal (adapted, defensive)
    # ---------------------------
    def verify_goal(self, goal_statement, debug=False):
        """
        Verify if a goal statement is consistent with the ontology.

        @param goal_statement: A GoalStatement object representing the goal of an actor in the world.
        @param debug: Verbose output
        @return: True or False
        """
        assert isinstance(goal_statement, GoalStatement), "Input must be a GoalStatement."

        # History check
        if goal_statement in self.history.get('valid', []):
            goal_statement.valid = True
            return True
        if goal_statement in self.history.get('invalid', []):
            goal_statement.valid = False
            return False

        # Find ontology individuals
        actor = self.find_individual(goal_statement.actor)
        goal = self.find_individual(goal_statement.goal.lower())
        target = self.find_individual(goal_statement.target)

        if debug:
            print("[KNOWLEDGE BASE] verify_goal:", actor, goal, target)

        if actor is None or goal is None or (target is not None and not self._is_known_individual(target)):
            if debug:
                print("Actor, goal, or target not found in ontology.")
            goal_statement.valid = False
            return False

        # Run under IPC lock
        lock = self._get_ipc_lock()
        with lock:
            try:
                # Temporary assignments
                setattr(actor, "has_goal", goal)
                setattr(goal, "involves_object", target)

                # Run reasoner
                goal_statement.valid = self._verify(engine="pellet", infer=False, debug=debug)

            finally:
                # Reset temporary properties
                try:
                    setattr(actor, "has_goal", None)
                    setattr(goal, "involves_object", None)
                except Exception:
                    if debug:
                        print("Warning: failed to reset goal temporary properties.")

        if debug:
            print("[KNOWLEDGE BASE] Goal consistent?", goal_statement.valid)
        return goal_statement.valid
    
    def cleanup_ontology(self):
        try:
            default_world.close()
            print("Ontology SQLite closed cleanly.")
        except Exception as e:
            print("Failed to close ontology cleanly:", e)
    
kb = KnowledgeBase('kitchen_onto')
