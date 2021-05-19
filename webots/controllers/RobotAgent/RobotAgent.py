"""
RobotAgent controller.
Derives from Agent.

TIAGo++: https://www.cyberbotics.com/doc/guide/tiagopp?version=master
"""
import math
import time

from Agent import Agent
from controller import CameraRecognitionObject, Field, InertialUnit, Motion
import numpy as np
import os
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace

MOTION_DIR = "motions"
MOTION_EXT = ".motion"
agents = ["HumanAgent"]                     # Humans and other robots will initially be unknown
# Some environmental elements and the robot itself are not considered
excluded = ["CircleArena", "myTiago++", "SolidBox", "Window", "Wall", "Cabinet", "Floor", "Ceiling", "Door",
            "RoCKInShelf"]
included = ["coca-cola", "human", "table(1)"]


class RobotAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        # World knowledge
        self.last_timestep = -1             # Last discretized timestep in which activity was recorded
        self.world_knowledge = {}           # Coordinates of all the entities in the world at present
        self.initialize_world_knowledge()
        self.qsrlib = QSRlib()                 # Qualitative Spatial Relationship engine
        self.world = World_Trace()          # Time-series coordinates of all the entities in the world
        self.update_world_trace()

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
            #if node_type_name not in excluded:
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
            if name in included:
                old_position = self.world_knowledge[name]
                if current_position != old_position:
                    self.world_knowledge[name] = current_position
                    if debug:
                        print("Updated the position of {0} to {1}".format(name, current_position))
        # Also updates the world trace
        self.update_world_trace(debug=debug)

    def update_world_trace(self, debug=False):
        """
        Initializes or updates the QSR world trace.

        :return: None
        """
        current_timestep = int(self.supervisor.getTime())
        if current_timestep > self.last_timestep:
            for name, position in self.world_knowledge.items():
                if position is not None:
                    # todo add xsize, ysize, zsize?
                    new_os = Object_State(name=name, timestamp=current_timestep,
                                          x=position[0], y=position[1], z=position[2])
                    self.world.add_object_state(new_os)
                    if debug:
                        print("Added {0} to world trace in position [{1}, {2}, {3}] at timestamp {4}.".format(
                            new_os.name, new_os.x, new_os.y, new_os.z, new_os.timestamp))
            self.last_timestep = current_timestep

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

    def rotate(self, degrees, direction, speed=0.5):
        """
        Rotates for a certain amount of degrees, left or right. Synchronous.

        :param degrees: objective degrees
        :param direction: 'left' or 'right'
        :param speed: float in (0.0, 1.0]
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
        rad = math.radians(degrees)
        while self.step():
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
                        self.rotate(45, "left")
                    else:
                        self.rotate(45, "right")
                self.motors['head_1'].setPosition(new_position)
            break

    def search_for(self, target_name):
        """
        Searches for a specific target node, rotating up to 360° to find it.

        :param target_name: Node name
        :return: True if found, False if not found
        """
        degrees = 0
        while self.step():
            # Observe the environment for the specific target
            objects = self.observe()
            self.update_world_knowledge(objects)
            target = self.get_object_from_set(target_name, objects)
            if target is None:
                # If the target is not found, rotate 90° and keep searching
                self.rotate(45, "right")
                degrees += 90
            else:
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
        file = os.path.join(MOTION_DIR, motion_name)
        file += MOTION_EXT
        if os.path.exists(file) and os.path.isfile(file):
            motion = Motion(file)
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

    def compute_qsr_test(self):
        # TODO experimental
        which_qsr = ["argd", "qtcbs"]
        dynamic_args = {
            "for_all_qsrs": {
                #"qsrs_for": [("human", "coca-cola"), ("human", "table(1)")]
                "qsrs_for": [("human", "coca-cola")]
            },
            "argd": {
                "qsr_relations_and_values": {"touch": 1, "near": 2, "medium": 4, "far": 8}
            },
            "qtcbs": {
                "quantisation_factor": 0.01,
                "validate": False,
                "no_collapse": True
            },
            "qstag": {
                "params": {"min_rows": 1, "max_rows": 1, "max_eps": 3},
                "object_types": {"human": "Human",
                                "coca-cola": "Coke"},
                                #"table(1)": "Table"}
            }
        }
        qsrlib_request_message = QSRlib_Request_Message(which_qsr, self.world, dynamic_args=dynamic_args)
        qsrlib_response_message = self.qsrlib.request_qsrs(req_msg=qsrlib_request_message)
        self.pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message)
        return qsrlib_response_message

    def pretty_print_world_qsr_trace(self, which_qsr, qsrlib_response_message):
        print(which_qsr, "request was made at ", str(qsrlib_response_message.req_made_at)
              + " and received at " + str(qsrlib_response_message.req_received_at)
              + " and finished at " + str(qsrlib_response_message.req_finished_at))
        print("---")
        print("Response is:")
        for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
            foo = str(t) + ": "
            for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),
                            qsrlib_response_message.qsrs.trace[t].qsrs.values()):
                foo += str(k) + ":" + str(v.qsr) + "; "
            print(foo)


# MAIN LOOP

robot = RobotAgent()
tracked_objects = ['can', 'pedestrian']

# Perform simulation steps until Webots is stopping the controller
#robot.motion("neutral")
#next_test = 30.0
while robot.step():
    if robot.is_camera_active():
        if robot.search_for("pedestrian"):
            robot.track_target("pedestrian")
            #if robot.supervisor.getTime() >= next_test:
                #next_test *= 2
                #robot.compute_qsr_test()
            if robot.supervisor.getTime() > 30:
                qsr_response = robot.compute_qsr_test()
                print(qsr_response.qstag.episodes)
        else:
            print("I didn't find the human!")
