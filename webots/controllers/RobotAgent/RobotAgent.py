"""
RobotAgent controller.
Derives from Agent.

TIAGo++: https://www.cyberbotics.com/doc/guide/tiagopp?version=master
"""
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


import math
import cProfile
import time

from Agent import Agent
from controller import Field, InertialUnit, Motion, Node
import os

from maps.Kitchen2 import Kitchen2
from qsrlib_io.world_trace import Object_State
from cognitive_architecture.Episode import *
from cognitive_architecture.CognitiveArchitecture import CognitiveArchitecture
from util.PathProvider import path_provider
from cognitive_architecture.QSRFactory import QSRFactory
from multiprocessing import Event
from datatypes.Synchronization import QSRSynch, SynchVariable
from controller import GPS, Compass

agents = ["Pedestrian"]   # Humans and other robots will initially be unknown -- 2021a was HumanAgent
# Some environmental elements and the robot itself are not considered
included = ["human", "sink", "glass", "hobs", "biscuits", "meal", "plate", "bottle"]

current_map = Kitchen2()


class RobotAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        # World knowledge
        self.last_timestep = -1             # Last discrete timestep in which activity was recorded
        self.world_knowledge = {}           # Coordinates of all the entities in the world at present
        self.initialize_world_knowledge()
        # QSR factory and synchronization mechanisms
        qsr_synch = QSRSynch()
        self.qsr_factory = QSRFactory(qsr_synch)
        # Set up the cognitive architecture process and its communication channels
        self.ca_conn = SynchVariable()
        self.start_event = Event()
        self.cognition = CognitiveArchitecture(self.ca_conn, qsr_synch, self.start_event, "TEST", verification=True)

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
        self.max_speed = 10.0
        self.inertial = InertialUnit("inertial unit")
        #self.compass = Compass("compass")

        # Enables and disables devices
        self.inertial.enable(self.timestep)
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        self.rangefinder.disable()
        #self.compass.enable(self.timestep)

        self.update_world_trace()
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
            id = object.getId()
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
                if typename == "Pedestrian":# was HumanAgent
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
                #print(name, position)
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
                                              x=position[0], y=position[1], hold=hold, label=training_label,
                                              target=training_target, ov=ov)
                    else:
                        new_os = Object_State(name=name, timestamp=current_timestep, x=position[0], y=position[1])
                    new_objects.append(new_os)
                    if debug:
                        print("Added {0} to world trace in position [{1}, {2}] at timestamp {3}.".format(
                            new_os.name, new_os.x, new_os.y, new_os.timestamp))
            # Sends the new observations to the cognitive architecture
            self.qsr_factory.add_observation(new_objects)
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
   
        # Rotation matrix (3×3)
        R = np.array(human.getOrientation()).reshape(3, 3)

        # Translation: Z is now up
        T = np.array([human_position[0], human_position[1], 1.27])

        # Local forward direction (in the new coordinate system, forward = +Y)
        p_local = np.array([1, 0, 0], dtype=float)#[0, 1, 0]

        # Transform to global coordinates
        p_global = np.dot(R, p_local) + T

        # Project to ground plane (X–Y)
        new_position = [round(p_global[0], 2), round(p_global[1], 2)]

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
        assert 0 <= degrees <= 360
        max_speed = 10.1523
        # If the angle is too narrow, skip these computations
        if 0 <= degrees <= 2:
            return
        if direction == "LEFT":
            left_velocity = -speed * max_speed
            right_velocity = speed * max_speed
        else:
            left_velocity = speed * max_speed
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
        # Observe the environment for the specific target
        objects = self.observe()
        self.update_world_knowledge(objects, debug=False)
        target = self.get_object_from_set(target_name, objects)
        if target is not None:
            # Calculate the distance from the center of the camera
            position_on_image = target.getPositionOnImage()
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

    def search_for(self, target_name):
        """ca_conn
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

    def get_robot_heading_Webots2021a(self):
        """
        DEPRECATED
        Using the internal compass, calculates the angle between the robot's facing and the world North.

        compass-style

        @return: Angle between the robot and the world's north, in degrees.
        """
        north = self.compass.getValues() # [Forward, Left, Up]
        rad = math.atan2(north[0], north[1])  # swapped order → 0°=North, CW positive
        heading = math.degrees(rad - 1.57)
        if heading < 0:
            heading += 360.0
        return heading
    
    def get_robot_heading(self):
        """
        Calculates the angle between the robot's facing direction
        and the world's North, using the InertialUnit (yaw angle).
        Returns heading in degrees [0, 360).

        0° = North, 90° = East, increases clockwise.

        Robot facing	Heading (deg)
            North (Y+)	    0°
            East (X+)	    90°
            South (Y−)	    180°
            West (X−)	    270°
        """
        roll, pitch, yaw = self.inertial.getRollPitchYaw()
        # Webots yaw is in radians, counterclockwise from world X axis
        heading = math.floor(math.degrees(yaw))

        # Convert to compass convention (0° = North, increasing clockwise) -- # Adjust: yaw=0 → East, so rotate to make 0 = North
        heading = (90 - heading) % 360
 
        return heading
        

    def motor_reset(self):
        """
        Resets the wheels velocity and position.

        @return: None
        """
        start = time.time()
        while self.step() and (time.time() - start) < 1:
            self.motors['wheel_left'].setPosition(float('inf'))
            self.motors['wheel_right'].setPosition(float('inf'))
            self.motors['wheel_left'].setVelocity(0.0)
            self.motors['wheel_right'].setVelocity(0.0)

    def walk(self, target, speed=1, debug=False):
        """
        Walks to a specific coordinate, without any animation.

        :param target: coordinate in the format (x, y)
        :param speed: walking speed in [m/s]
        :param debug: activate/deactivate debug output
        :return: None
        """
        start_position = self.get_robot_position()
        print(start_position, target)
        if speed is None:
            speed = self.speed
        end_position = target
        # ROTATION
        self.motor_reset()
        # Calculates the robot's heading
        heading = self.get_robot_heading()
        # Calculates the angle between the robot's position and the target
        dx = end_position[0] - start_position[0] # North component
        dy = end_position[1] - start_position[1] # East component
        destination_angle = math.degrees(math.atan2(dx, dy)) # using the robot frame (because how we use heading)
        # Calculates the angle between the robot's current heading and the destination
        angle = destination_angle - heading
        print("Current heading: {0}°. Destination angle: {1}°. Need to rotate {2}°.".format(
            round(heading, 2), destination_angle, round(angle, 2)))
        if angle > 180.0:
            angle -= 360.0 - angle
        elif angle < -180.0:
            angle = 360.0 + angle
        print("Normalized rotation angle: {0}°.".format(round(angle, 2)))
        # Performs the rotation
        if angle > 0:
            self.rotate(angle, "RIGHT", speed=0.1)
            print("Rotating {0}° to the RIGHT".format(angle))
        else:
            self.rotate(-angle, "LEFT", speed=0.1)
            print("Rotating {0}° to the LEFT".format(angle))
        # MOVEMENT
        self.motor_reset()
        current_position = self.get_robot_position()
        total_distance = math.dist(current_position, end_position)
        print("Walking towards {0} from {1}, total distance {2} m.".format(
            end_position, current_position, round(total_distance, 2)))
        while self.step():
            remaining_distance = math.dist(self.get_robot_position(), end_position)
            if remaining_distance > 0.5:
                # Speed decreases linearly with proximity
                velocity = self.max_speed * (remaining_distance / total_distance)
                self.motors['wheel_left'].setVelocity(velocity)
                self.motors['wheel_right'].setVelocity(velocity)
            else:
                break
        self.motor_reset()

    def approach_target(self, target_name, speed=None, debug=False):
        """
        Supervisor-version of search-and-approach: uses global knowledge of the world to identify the target coordinates
        without needing to visually identify the destination.

        :param target_name: str, name of the desired object
        :return: True if successful, False otherwise
        """
        assert isinstance(target_name, str), "Must specify a string name to search for"
        target = self.all_nodes.get(target_name)
        if target is not None:
            # If identified, gets its world position
            target_position = self.convert_to_2d_coords(target.getPosition())
            print("Target {0} found at {1}.".format(target_name, target_position))
            # Trace a path to the destination
            print("Path planning, please wait...")
            path = self.path_planning(current_map, target_position, show=True)
            print("Path found!")
            if path is not None:
                print(path)
                for waypoint in path:
                    self.walk(waypoint, speed=speed, debug=False)
                    print("Reached waypoint {0}.".format(waypoint))
                    print("-----------------------------------------------------------------------------------------------")
                self.turn_towards(target_position)
                return True
            else:
                print("Couldn't path plan to {0}!".format(target_name))
        else:
            print("Couldn't find target: {0}".format(target_name))
        return False

    def approach_coordinates(self, coordinates, speed=None, debug=False):
        """
        Same as approach_target, but moves to certain coordinates instead of referencing a node in the environment.

        :param coordinates: (X,Y) tuple
        :return: True if successful, False otherwise
        """
        assert isinstance(coordinates, tuple), "Must specify a tuple (X,Y) to walk to"
        # Trace a path to the destination
        print("Path planning, please wait...")
        path = self.path_planning(current_map, coordinates, show=False)
        print("Path found!")
        if path is not None:
            print("Walking towards {0}".format(coordinates))
            for waypoint in path:
                self.walk(waypoint, speed=speed, debug=False)
            self.turn_towards(coordinates)
            return True
        else:
            print("Couldn't path plan to {0}!".format(coordinates))
        return False

    def initialize(self):
        # Neutral position
        self.motion("neutral")
        # Start the brain
        self.cognition.start()
        # Wait for the ok to start the subprocesses of cognition
        self.cognition.start_event.wait()
        # Fully boot the cognition
        self.cognition.lowlevel.start()
        self.cognition.highlevel.start()

    def main(self):
        self.initialize()
        while self.step():
            # Fist of all, check if a goal was inferred
            if self.ca_conn.poll():
                plan = self.ca_conn.get()
                print("I am now ready to act. This is what I will do:")
                for action in plan:
                    print("{0}{1}".format(action.name, action.parameters))
            elif self.is_camera_active():
                if self.search_for("pedestrian"):
                    self.track_target("pedestrian")
                else:
                    print("I didn't find the human!")

    def get_robot_orientation(self):
        """
        Using Supervisor functions, obtains the rotation matrix of the robot.

        :return: 3x3 rotation matrix
        """
        orientation = np.array(self.supervisor.getSelf().getOrientation())
        orientation = orientation.reshape(3, 3)
        return orientation


# MAIN LOOP
def main():
    #map = Kitchen2()
    robot = RobotAgent()
    #robot.initialize()
    robot.main()
    #robot.walk_to(target=(2.25851, 1.59772), speed=1)
    #robot.motion("neutral")
    #robot.walk_to((1.0, 1.0), speed=1, debug=True)
    #robot.walk_to((3.37, 2.0), speed=1, debug=True)
    #robot.approach_target('plate')
    #robot.rotate(90.0, "RIGHT", speed=1, observe=False)
    #robot.rotate(90.0, "LEFT", speed=1, observe=False)
    #robot.rotate(90.0, "RIGHT", speed=1, observe=False)
    #print("Hello!")

# Run this code to benchmark execution time
#cProfile.run('main()', sort='time')
main()
