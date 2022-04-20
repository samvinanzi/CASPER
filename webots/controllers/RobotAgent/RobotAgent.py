"""
RobotAgent controller.
Derives from Agent.

TIAGo++: https://www.cyberbotics.com/doc/guide/tiagopp?version=master
"""

import math
import pickle
import cProfile
from Agent import Agent
from controller import Field, InertialUnit, Motion, Node
import os
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State
from cognitive_architecture.Episode import *
from cognitive_architecture.CognitiveArchitecture import CognitiveArchitecture
from util.PathProvider import path_provider

agents = ["HumanAgent"]                     # Humans and other robots will initially be unknown
# Some environmental elements and the robot itself are not considered
included = ["human", "sink", "glass", "hobs", "biscuits", "meal", "plate", "bottle"]


class RobotAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        # World knowledge
        self.last_timestep = -1             # Last discretized timestep in which activity was recorded
        self.world_knowledge = {}           # Coordinates of all the entities in the world at present
        self.initialize_world_knowledge()
        self.qsrlib = QSRlib()                 # Qualitative Spatial Relationship engine
        #self.world_trace = World_Trace()          # Time-series coordinates of all the entities in the world
        #self.update_world_trace()

        # todo !
        self.cognition = CognitiveArchitecture(mode="TEST")
        self.cognition.start()
        self.update_world_trace()
        # todo !

        # Devices
        self.motors = {'head_1': self.supervisor.getDevice("head_1_joint"),
                       'head_2': self.supervisor.getDevice("head_2_joint"),
                       'torso': self.supervisor.getDevice("torso_lift_joint"),
                       'arm_right_1': self.supervisor.getDevice("arm_right_1_joint"),
                       'arm_right_2': self.supervisor.getDevice("arm_right_2_joint"),
                       'arm_right_3': self.supervisor.getDevice("arm_right_3_joint"),
                       'arm_right_4': self.supervisor.getDevice("arm_right_4_joint"),
                       'arm_right_5': self.supervisor.getDevice("arm_right_5_joint"),
                       'arm_right_6': self.supervisor.getDevice("arm_right_6_joint"),
                       'arm_right_7': self.supervisor.getDevice("arm_right_7_joint"),
                       'arm_left_1': self.supervisor.getDevice("arm_left_1_joint"),
                       'arm_left_2': self.supervisor.getDevice("arm_left_2_joint"),
                       'arm_left_3': self.supervisor.getDevice("arm_left_3_joint"),
                       'arm_left_4': self.supervisor.getDevice("arm_left_4_joint"),
                       'arm_left_5': self.supervisor.getDevice("arm_left_5_joint"),
                       'arm_left_6': self.supervisor.getDevice("arm_left_6_joint"),
                       'arm_left_7': self.supervisor.getDevice("arm_left_7_joint"),
                       'wheel_left': self.supervisor.getDevice('wheel_left_joint'),
                       'wheel_right': self.supervisor.getDevice('wheel_right_joint')
                       }
        self.inertial = InertialUnit("inertial unit")

        # Enables and disables devices
        self.inertial.enable(self.timestep)
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        self.rangefinder.disable()

        print(str(self.__class__.__name__) + " has activated.")

    def initialize_world_knowledge(self):
        """
        Initializes the world knowledge: static items have their initial coordinates, agents are unknown, environmental
        objects are ignored.

        :return: None
        """
        for name, node in self.all_nodes.items():
            node_type_name = node.getTypeName()  # Elements are filtered by type, not by name
            if name in included:
                if node_type_name in agents:
                    position = None
                else:
                    position = node.getPosition()
                self.world_knowledge[name] = position

    def update_world_knowledge(self, objects, debug=False):
        """
        Real-time update of the world knowledge. Updates the coordinates of the observed objects.
        This function should be called everytime the robot invokes observe().

        :param objects: list of CameraRecognitionObjects
        :return: None
        """
        for object in objects:
            # Uses the id of the recognized object to retrieve its node in the scene tree
            id = object.get_id()
            node = self.supervisor.getFromId(id)
            current_position = node.getPosition()
            name_field = node.getField("name")
            name = name_field.getSFString()
            typename = node.getTypeName()
            held_objects_corrections = {}
            if name in included:
                old_position = self.world_knowledge[name]
                if current_position != old_position:
                    self.world_knowledge[name] = current_position
                    if debug:
                        print("Updated the position of {0} to {1}".format(name, current_position))
                # If the node is the human, check if they are holding items in their hands. If this is the case,
                # prepare a list of corrections (they must be registered at the same position of the human)
                if typename == "HumanAgent":
                    object_held_field: Field = node.getField("heldObjectReference")
                    object_held_id = object_held_field.getSFInt32()
                    if object_held_id != 0:
                        held_objects_corrections[object_held_id] = current_position
                # Go through the corrections
                for key, value in held_objects_corrections.items():
                    node_to_correct = self.supervisor.getFromId(key)
                    name_field = node_to_correct.getField("name")
                    name = name_field.getSFString()
                    self.world_knowledge[name] = value
        # Also updates the world trace
        self.update_world_trace(debug=debug)

    def update_world_trace(self, debug=False):
        """
        Initializes or updates the QSR world trace.

        :return: None
        """
        current_timestep = int(self.supervisor.getTime())
        if current_timestep > self.last_timestep:
            new_objects = []
            for name, position in self.world_knowledge.items():
                if position is not None:
                    # If we are dealing with a human, let's register their specific states
                    if name == "human":
                        node = self.all_nodes.get(name)
                        # HOLD state
                        object_held_field: Field = node.getField("heldObjectReference")
                        object_held_id = object_held_field.getSFInt32()
                        hold = object_held_id != 0
                        # Training label (if it exists)
                        training_label_field: Field = node.getField("trainingTaskLabel")
                        training_label = training_label_field.getSFString()
                        # Training target
                        training_target_field: Field = node.getField("trainingTaskTarget")
                        training_target = training_target_field.getSFString()
                        # Adjustment for PICK action
                        if training_label == "PICK":
                            hold = False
                        # Calculates the orientation vector, based on the human position and rotation
                        ov = self.calculate_human_orientation_vector(name, (position[0], position[1]))
                        # ObjectState insertion
                        new_os = Object_State(name=name, timestamp=current_timestep,
                                              x=position[0], y=position[2], hold=hold, label=training_label,
                                              target=training_target, ov=ov)
                    else:
                        new_os = Object_State(name=name, timestamp=current_timestep, x=position[0], y=position[2])
                    new_objects.append(new_os)
                    if debug:
                        print("Added {0} to world trace in position [{1}, {2}] at timestamp {3}.".format(
                            new_os.name, new_os.x, new_os.y, new_os.timestamp))
            # Sends the new observations to the cognitive architecture
            self.cognition.lowlevel.tq.add_observation(new_objects)
            self.last_timestep = current_timestep

    def calculate_human_orientation_vector(self, human_name, human_position):
        """
        Calculates the orientation vector for a specific human

        :param human_name: Name of the desired human, str
        :param human_position: Tuple representing the (x,y) position of the human
        :return np.array orientation vector
        """
        assert isinstance(human_name, str), "Must specify a string name to search for"
        human: Node = self.all_nodes.get(human_name)
        if human is None:
            return None
        R = np.array(human.getOrientation())
        R = R.reshape(3, 3)
        T = np.array([human_position[0], 1.27, human_position[1]])
        p_local = np.array([0, 0, 1], dtype=float)
        p_global = np.dot(R, p_local) + T
        new_position = [round(p_global[0], 2), round(p_global[2], 2)]
        dx = human_position[0] - new_position[0]
        dy = human_position[1] - new_position[1]
        return np.asarray([dx, dy])

    def get_target_coordinates(self, target_name):
        """
        Retrieves the global coordinates of a specified node.

        :param target_name: Name of the desired node.
        :return: (x,y,z) coordinates
        """
        assert isinstance(target_name, str), "Must specify a string name to search for"
        target = self.all_nodes.get(target_name)
        if target is not None:
            # If identified, gets its world position
            target_position = target.getPosition()
            return target_position

    def rotate(self, degrees, direction, speed=0.5, observe=True):
        """
        Rotates for a certain amount of degrees, left or right. Synchronous.

        :param degrees: objective degrees
        :param direction: 'left' or 'right'
        :param speed: float in (0.0, 1.0]
        :param observe: if True, keeps observing the environment while rotating
        :return: None
        """
        direction = direction.upper()
        assert direction == "LEFT" or direction == "RIGHT", "Direction must be 'left' or 'right'."
        assert 0.0 < speed <= 1.0, "Speed must be in (0, 1]."
        assert 0 < degrees <= 360
        max_speed = 10.1523
        if direction == "LEFT":
            left_velocity = -speed * max_speed
            right_velocity = 0.0
        else:
            left_velocity = 0.0
            right_velocity = -speed * max_speed
        # Motor initialization
        self.motors['wheel_left'].setPosition(float('inf'))
        self.motors['wheel_right'].setPosition(float('inf'))
        self.motors['wheel_left'].setVelocity(0.0)
        self.motors['wheel_right'].setVelocity(0.0)
        initial = self.inertial.getRollPitchYaw()
        rad = math.radians(degrees)     # Objective rotation
        while self.step():
            if observe:
                self.update_world_knowledge(self.observe())
            current = self.inertial.getRollPitchYaw()
            if math.fabs(initial[2] - current[2]) < rad:
                self.motors['wheel_left'].setVelocity(left_velocity)
                self.motors['wheel_right'].setVelocity(right_velocity)
            else:
                self.motors['wheel_left'].setVelocity(0.0)
                self.motors['wheel_right'].setVelocity(0.0)
                break

    def track_target(self, target_name):
        """
        Visually tracks a target in the environment.

        :param target_name: name of the node to follow.
        :return: None
        """
        while self.step():
            # Observe the environment for the specific target
            objects = self.observe()
            self.update_world_knowledge(objects, debug=False)
            target = self.get_object_from_set(target_name, objects)
            if target is not None:
                # Calculate the distance from the center of the camera
                position_on_image = target.get_position_on_image()
                error = self.camera.getWidth() / 2 - position_on_image[0]
                current_position = self.motors['head_1'].getTargetPosition()    # Head motor position
                max_position = self.motors['head_1'].getMaxPosition()           # Head motor max rotation
                # Minimum distance of the target from the center of the camera to initialize movement
                deadzone = self.camera.getWidth() / 10
                if math.fabs(error) > deadzone:
                    diff = 0.02
                    if error < 0:
                        diff *= -1
                else:
                    diff = 0.0
                new_position = current_position + diff
                if math.fabs(new_position) > max_position:
                    new_position = 0.0
                    if error > 0:
                        self.rotate(10.0, "left", observe=True)
                    else:
                        self.rotate(10.0, "right", observe=True)
                self.motors['head_1'].setPosition(new_position)
            #break

    def search_for(self, target_name):
        """
        Searches for a specific target node, rotating up to 360° to find it.

        :param target_name: Node name
        :return: True if found, False if not found
        """
        degrees = 0
        frames_no_target = 0    # Number of frames without a clear target (used to decide when to rotate and search)
        while self.step():
            # Observe the environment for the specific target
            objects = self.observe()
            self.update_world_knowledge(objects)
            target = self.get_object_from_set(target_name, objects)
            if target is None:
                frames_no_target += 1
                print("No target, {0}".format(frames_no_target))
                # This should refrain the robot from rotating if it looses sight of the human for just a moment, for
                # example during a temporary occlusion.
                if frames_no_target >= 100:
                    frames_no_target = 0
                    # If the target is not found, rotate 90° and keep searching
                    self.rotate(45, "right")
                    degrees += 90
            else:
                frames_no_target = 0
                return True     # The target is found
            if degrees >= 360:
                # After a 360° turn, the target was not found, so it must not be in range
                return False

    def motion(self, motion_name, debug=False):
        """
        Executes a pre-scripted motion and waits until its completion.

        :param motion_name: Name of the motion file (without base path or extension)
        :param debug: Activates debug output
        :return: True if completed, False otherwise
        """
        file = path_provider.get_robot_motion(motion_name)
        if os.path.exists(file) and os.path.isfile(file):
            motion = Motion(str(file))
            if debug:
                print("Loading motion: {0}".format(file))
            if motion.isValid():
                # If another motion was in progress, interrupt it
                if self.currentlyPlaying:
                    if debug:
                        print("Interrupting previously started motion.")
                    self.currentlyPlaying.stop()
                # Start moving
                motion.play()
                self.currentlyPlaying = motion
                # Wait for the movement to end
                while not motion.isOver():
                    self.step()
                self.currentlyPlaying = None
                return True
            else:
                print("[ERROR] File {0} does not contain a valid motion.".format(file))
                return False
        else:
            print("[ERROR] Invalid motion name: {0}".format(motion_name))


# MAIN LOOP

def main():
    robot = RobotAgent()
    # Perform simulation steps until Webots is stopping the controller
    robot.motion("neutral")
    while robot.step():
        if robot.is_camera_active():
            if robot.search_for("pedestrian"):
                robot.track_target("pedestrian")
                #if robot.supervisor.getTime() >= 40:
                    #qsr_response = robot.compute_qsr_test()
                    #robot.cognition.lowlevel.compute_qsr(show=True)
                    #pickle.dump(qsr_response, open(os.path.join(BASEDIR, "data\pickle\qsr_response{0}.p".format(i)), "wb"))
                    #pickle.dump(robot.world_trace, open(os.path.join(BASEDIR, "data\pickle\world_trace{0}.p".format(i)), "wb"))
                    #print("Saved")
                #    break
            else:
                print("I didn't find the human!")

# Run this code to benchmark execution time
cProfile.run('main()', sort='time')
#main()

