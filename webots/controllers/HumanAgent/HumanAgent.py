"""
HumanAgent controller.
Derives from Agent.
"""

UPPER_ARM_LENGTH = 0.25  # m (estimate)
LOWER_ARM_LENGTH = 0.23  # m (estimate)
HAND_LENGTH = 0.10       # m (estimate)

import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Agent import Agent
from controller import Node, Field, Proto
import math
import numpy as np
from maps.Kitchen2 import Kitchen2
from copy import copy
import cProfile
from typing import Tuple

current_map = Kitchen2()


class HumanAgent(Agent):
    def __init__(self, mode="TRAIN", debug=True):
        Agent.__init__(self, agentName = "Human", debug=debug)

        assert mode.upper() == "TRAIN" or mode.upper() == "TEST", "mode must be TEST or TRAIN."
        self.mode = mode
        # Enables and disables devices
        self.camera.disable()
        #self.camera.recognitionEnable(self.timestep)
        self.rangefinder.disable()

        # Walking parameters
        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        self.joints_position_field = []
        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]

        # Populates joints_position_field with the nodes
        root_node_ref = self.supervisor.getSelf()
        for i in range(0, self.BODY_PARTS_NUMBER):
            joint = root_node_ref.getField(self.joint_names[i])
            self.joints_position_field.append(joint)

        self.height_offsets = [
            # those coefficients are empirical coefficients which result in a realistic walking gait
            -0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03
        ]
        self.angles = [  # those coefficients are empirical coefficients which result in a realistic walking gait
            [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],  # left arm
            [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],  # left lower arm
            [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],  # left hand
            [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],  # right arm
            [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38],  # right lower arm
            [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],  # right hand
            [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],  # left leg
            [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],  # left lower leg
            [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],  # left foot
            [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],  # right leg
            [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],  # right lower leg
            [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],  # right foot
            [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09]  # head
        ]
        
        # Resets the object held field (doesn't reset automatically between Webots executions)
        object_held_field: Field = self.supervisor.getSelf().getField("heldObjectReference")
        object_held_field.setSFInt32(0)

        self.startingPosition = root_node_ref.getField("translation").getSFVec3f()

        print(str(self.__class__.__name__) + " has activated.")

    def get_in_hand_name(self):
        """
        Retrieves the name of the handheld object.

        :return: str name.
        """
        if self.object_in_hand is not None:
            return self.object_in_hand.getField("name").getSFString()
        else:
            return None

    def turn_towards(self, target):
        """
        Rotates the robot, in place, towards the target.

        :param target: coordinate in the format (x, y)
        :return: None
        """
        start = self.get_robot_position()
        end = target
        root_node_ref = self.supervisor.getSelf()
        root_rotation_field = root_node_ref.getField("rotation")
        # Rotate
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dx, dy)
        while self.step():
            root_rotation_field.setSFRotation([0, 0, 1, angle])
            break

    def walk(self, target, speed=None, enable_bumper=False, debug=False):
        """
        Walks to a specific coordinate, without any animation.

        :param target: coordinate in the format (x, y)
        :param speed: walking speed in [m/s]
        :param enable_bumper: Activates or deactivates the bumper collision detection
        :param debug: activate/deactivate debug output
        :return: None
        """
        if speed is None:
            speed = self.speed
        # Calculates the waypoints (including the starting position)
        start = self.get_robot_position()
        end = target
        if debug:
            print("I am about to walk from {0} to {1}".format(start, end))
        # Retrieve the required nodes and fields
        root_node_ref = self.supervisor.getSelf()  # root_node_ref is the Robot node where the Supervisor is running
        root_translation_field = root_node_ref.getField("translation")
        root_rotation_field = root_node_ref.getField("rotation")

        # Rotate
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dy, dx) # Using the world frame
        if debug:
            print("Angle: {0} rad ({1}°)".format(angle, math.degrees(angle)))
            print("Distance: {0}".format(math.dist(start, end)))
        root_rotation_field.setSFRotation([0, 0, 1, angle])

        # Move
        s = np.array(start)
        e = np.array(end)
        # Line segment: f(0) = start, f(1) = end
        f = lambda t: s + t * (e - s)

        delta = 0.01 * speed    # When speed = 1, it ensures a movement of 1 m/s
        i = 0
        while self.step():
            if enable_bumper:
                collision = self.is_bumper_pressed()
            else:
                collision = False
            if not collision:
                # Steps forward and memorizes the new position
                i += delta
                waypoint = f(i)
            else:
                # Step backwards to avoid occupying the intersecting space (which will result in further detections)
                waypoint = f(i-delta)
            root_translation_field.setSFVec3f([waypoint[0], waypoint[1], self.ROOT_HEIGHT])
            # If an object is held, it has to move together with the agent
            self.move_object_in_hand()
            # Stop if the walk is complete or if an obstacle was hit
            if i >= 1 or collision:
                if debug:
                    print("I have stopped in position: {0}".format(waypoint))
                return

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
            # Trace a path to the destination
            print("Path planning, please wait...")
            path = self.path_planning(current_map, target_position, show=True)
            print("Path found!")
            if path is not None:
                print("Walking towards {0}".format(target_name))
                # Updates the training fields
                if self.object_in_hand:
                    self.update_training_label("TRANSPORT")
                    # If they are transporting, the target is the object, nor the destination
                    self.update_training_target(self.get_in_hand_name())
                else:
                    self.update_training_label("WALK")
                    self.update_training_target(target_name)
                for waypoint in path:
                    self.walk(waypoint, speed=speed, debug=False)        #todo re-enable if required
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
            # Updates the training fields
            if self.object_in_hand:
                self.update_training_label("TRANSPORT")
                # If they are transporting, the target is the object, nor the destination
                self.update_training_target(self.get_in_hand_name())
            else:
                self.update_training_label("WALK")
                #self.update_training_target(coordinates)
            for waypoint in path:
                self.walk(waypoint, speed=speed, debug=False)        #todo re-enable if required
            self.turn_towards(coordinates)
            return True
        else:
            print("Couldn't path plan to {0}!".format(coordinates))
        return False

    def neutral_position(self):
        """
        Positions the human agent in a neutral position.

        :return: None
        """
        while self.step():
            for i in range(0, self.BODY_PARTS_NUMBER):
                self.joints_position_field[i].setSFFloat(0.0)
            break

    def hand_forward(self, steps=5):
        """
        Adopts a position where the hand is leaning frontally. Animated.

        :param steps: int, desired number of animation waypoints
        :return: None
        """
        assert isinstance(steps, int) and steps >= 1, "Steps must be an integer not lower than 1."
        # Start positions
        upper_start = 0
        lower_start = 0
        # Target angles, defined empirically
        upper_target = -0.485398
        lower_target = -0.3
        # Increments per step
        upper_step = (upper_target - upper_start) / steps
        lower_step = (lower_target - lower_start) / steps
        # Indexes in the joint field list
        upper_arm_index = self.joint_names.index("rightArmAngle")
        lower_arm_index = self.joint_names.index("rightLowerArmAngle")
        # Incremental positions
        upper_angle = upper_start
        lower_angle = lower_start
        while self.step():
            if math.fabs(upper_target - upper_angle) > 0 and math.fabs(lower_target - lower_angle) > 0:
                self.joints_position_field[upper_arm_index].setSFFloat(float(upper_angle))
                self.joints_position_field[lower_arm_index].setSFFloat(float(lower_angle))
                upper_angle += upper_step
                lower_angle += lower_step
            else:
                break
            #self.step()
            #break

    def is_trainingmode(self):
        return True if self.mode == "TRAIN" else False

    def pick(self, target_name: str):
        """
        Positions the human in a grasping pose, then attaches the object (specified by its name) to the hand. Warning:
        the object is grasped even if it is not in range!

        :param target_name: str, name of the object to grasp
        :return: True if successful, False otherwise
        """
        assert isinstance(target_name, str), "Must specify a string name to search for"
        print("PICK {0}".format(target_name))
        self.hand_forward()
        target: Node = self.all_nodes.get(target_name)
        if target is not None:
            self.object_in_hand = target
            print("Object {0} grasped.".format(target_name))
            object_held_field: Field = self.supervisor.getSelf().getField("heldObjectReference")
            object_held_field.setSFInt32(target.getId())
            #self.move_object_in_hand()
            self.object_in_hand.resetPhysics()
            hand_translation, hand_rotation = self.get_hand_transform()
            object_rotation = self.object_in_hand.getField("rotation").getSFRotation()
            # Maintain the object's angle
            hand_rotation[-1] = object_rotation[-1]
            if self.is_trainingmode():
                self.busy_waiting(2, label="PICK", target=target_name)    # wait   
            self.update_training_label("PICK")
            self.update_training_target(target_name)
            while self.step():
                self.object_in_hand.getField("translation").setSFVec3f(hand_translation)
                self.object_in_hand.getField("rotation").setSFRotation(hand_rotation)
                break
            self.step()
            if self.is_trainingmode():
                self.busy_waiting(1, label="PICK", target=target_name)  # wait
            return True
        return False

    def place(self, destination_name):
        """
        Releases the hand-held object on a target destination (must be a surface)
        :param destination: str, name of the desired surface node
        :return: True if successful, False otherwise
        """
        assert isinstance(destination_name, str), "Must specify a string name to search for"
        print("PLACE {0} {1}".format(self.object_in_hand.getField("name").getSFString(), destination_name))
        if self.object_in_hand is not None:
            destination: Node = self.all_nodes.get(destination_name)
            if destination is not None:
                target_name = self.object_in_hand.getField("name").getSFString()
                # Verifies that the destination is an acceptable surface
                accepted_types = ["Table", "Plate", "Sink", "HotPlate", "Glass", "Worktop", "WaterBottle"]
                if destination.getTypeName() not in accepted_types:
                    print("Destination must be one of: {0} (selected: {1})".format(accepted_types, destination.getTypeName()))
                else:
                    """ Old position calculation, based on a tabletop. New one is based on destination object.
                    # The destination must have X and Y coordinates of the surface, Z equal to the table height
                    surface_trans = destination.getField("translation").getSFVec3f()
                    # Get the height of the table (Z)
                    surface_size = destination.getField("size").getSFVec3f()
                    surface_height = surface_size[2]
                    # Calculate the new position and assign it
                    final_position = [surface_trans[0], surface_trans[1]], surface_height]
                    """
                    surface_trans = destination.getField("translation").getSFVec3f()
                    final_position = copy(surface_trans)
                    if self.is_trainingmode():
                        self.busy_waiting(2, label="PLACE", target=target_name)     # wait
                    self.update_training_label("PLACE")
                    self.update_training_target(target_name)
                    self.object_in_hand.getField("translation").setSFVec3f(final_position)
                    #print("Deposited the {0} in {1}".format(self.object_in_hand.getTypeName(), final_position))
                    # Free the hand and reset the stance
                    self.object_in_hand = None
                    object_held_field: Field = self.supervisor.getSelf().getField("heldObjectReference")
                    object_held_field.setSFInt32(0)
                    if self.is_trainingmode():
                        self.busy_waiting(1, label="PLACE", target=target_name) # wait
                    self.neutral_position()
                    return True
        else:
            print("Can't find destination {0} to release object!".format(destination_name))
        return False

    def move_object_in_hand(self):
        """
        Called to update the position of the held object to match the one of the hand.

        :return: None
        """
        if self.object_in_hand is not None:
            # Resets the object inertia, preventing it from being subject to gravitational effects
            self.object_in_hand.resetPhysics()
            hand_translation, hand_rotation = self.get_hand_transform()
            object_rotation = self.object_in_hand.getField("rotation").getSFRotation()
            # Maintain the object's angle
            hand_rotation[-1] = object_rotation[-1]
            self.object_in_hand.getField("translation").setSFVec3f(hand_translation)
            self.object_in_hand.getField("rotation").setSFRotation(hand_rotation)
  

    def get_hand_transform(self):
        """
        Compute the world position and rotation of the right hand.
        """

        # Get human world pose
        robot = self.supervisor.getSelf()
        translation = np.array(robot.getField("translation").getSFVec3f())
        rotation_field = robot.getField("rotation").getSFRotation()

        # Segment lengths (for 1.82 m person)
        L1, L2, L3 = 0.35, 0.26, 0.19  # upper arm, lower arm, hand

        # Shoulder offset: from torso center to shoulder (in FLU)
        shoulder_offset_right = np.array([0.0, -0.20, 0.0])  # 20 cm to the right
        # (for left arm, use np.array([0.0, +0.20, 0.0]))

        # Angles (Webots fields, in radians)
        rightArmAngle = robot.getField("rightArmAngle").getSFFloat()
        rightLowerArmAngle = robot.getField("rightLowerArmAngle").getSFFloat()
        wrist_angle = robot.getField("rightHandAngle").getSFFloat()

        # Compute hand world pose
        p_world, R_world_hand = self.compute_hand_pose_world(
            translation, rotation_field,
            shoulder_offset_right,
            L1, L2, L3,
            rightArmAngle, rightLowerArmAngle, wrist_angle
        )

        hand_rotation = self.matrix_to_axis_angle(R_world_hand)
        #print("Hand center (world ENU):", p_world, hand_rotation)

        # Convert everything to standard Python floats and lists
        p_world_converted = [float(x) for x in p_world]
        hand_rotation_converted = [float(x) for x in hand_rotation]

        return p_world_converted, hand_rotation_converted
    
    def matrix_to_axis_angle(self, R):
        """
        Convert a 3×3 rotation matrix to Webots [axis_x, axis_y, axis_z, angle].
        """
        angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2.0)))
        if abs(angle) < 1e-6:
            # No rotation
            return [0.0, 1.0, 0.0, 0.0]
        rx = R[2,1] - R[1,2]
        ry = R[0,2] - R[2,0]
        rz = R[1,0] - R[0,1]
        axis = np.array([rx, ry, rz])
        axis /= np.linalg.norm(axis)
        return [axis[0], axis[1], axis[2], angle]


    def axis_angle_to_matrix(self, axis, angle):
        """Convert Webots axis-angle rotation to 3×3 rotation matrix."""
        axis = np.array(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        c = math.cos(angle)
        s = math.sin(angle)
        C = 1 - c
        return np.array([
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
        ])


    def rot_y(self, theta):
        """Rotation matrix around Y axis."""
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            [ c, 0,  s],
            [ 0, 1,  0],
            [-s, 0,  c]
        ])


    def compute_hand_local(self, L1, L2, L3, shoulder_angle, elbow_angle, wrist_angle):
        """
        Compute the hand position and rotation in the human local FLU frame.

        Conventions:
        - 0 = arm straight down (−Z)
        - Negative angles = arm lifts upward/forward (+X, +Z)
        - Positive angles = arm lowers backward
        """

        # Segment direction (downward)
        dir_segment = np.array([0.0, 0.0, -1.0])

        # Rotations around Y
        R1 = self.rot_y(shoulder_angle)
        R2 = self.rot_y(shoulder_angle + elbow_angle)
        R3 = self.rot_y(shoulder_angle + elbow_angle + wrist_angle)

        # Shoulder position (0.1 m above torso origin)
        p_shoulder = np.array([0.0, 0.0, 0.3])

        # Forward kinematics
        p_elbow = p_shoulder + L1 * (R1 @ dir_segment)
        p_wrist = p_elbow + L2 * (R2 @ dir_segment)
        p_hand  = p_wrist + 0.5 * L3 * (R3 @ dir_segment)

        p_local = p_hand
        R_local = R3
        return p_local, R_local


    def compute_hand_pose_world(self,
        human_translation, human_rotation_field,
        shoulder_offset_local_flu,
        L1, L2, L3, shoulder_angle, elbow_angle, wrist_angle
    ):
        """Compute the hand center position and rotation in world coordinates."""
        axis = human_rotation_field[:3]
        angle = human_rotation_field[3]
        R_world_human = self.axis_angle_to_matrix(axis, angle)

        # Local hand pose
        p_hand_local, R_hand_local = self.compute_hand_local(
            L1, L2, L3, shoulder_angle, elbow_angle, wrist_angle
        )

        # Add shoulder lateral offset
        p_total_local = shoulder_offset_local_flu + p_hand_local

        # Transform to world
        p_world = R_world_human @ p_total_local + np.array(human_translation)
        R_world_hand = R_world_human @ R_hand_local
        return p_world, R_world_hand
    

    def busy_waiting(self, duration, label="STILL", target="", debug=False):
        """
        Busy waiting, for a specified duration or infinitely.
        Overloaded method from Agent, in which the hand-held object's physics is resetted (not perfectly).

        :param duration: time to wait, in seconds. If -1, it loops infinitely.
        :param label: defines the training label for this activity.
        :param debug: debug output activation.
        :return: None
        """
        assert duration > 0 or duration == -1, "Duration has to be greater than 0, or exactly -1 for infinite waiting."
        self.update_training_label(label)
        self.update_training_target(target)
        if debug:
            print("{0} has gone asleep.".format(self.__class__.__name__))
        start = self.supervisor.getTime()
        if duration == -1:
            end = float('inf')
        else:
            end = start + duration
        while self.supervisor.getTime() < end:
            if self.object_in_hand:
                self.object_in_hand.resetPhysics()
            self.step()
            if self.object_in_hand:
                self.object_in_hand.resetPhysics()
        if debug:
            print("{0} has awoken.".format(self.__class__.__name__))

    def update_training_label(self, label="", visualize=False):
        """
        Updates the training task label.

        :param label: str, defaults to empty
        :param visualize: if True, prints a label on the Webots 3d view
        :return: None
        """
        label_field: Field = self.supervisor.getSelf().getField("trainingTaskLabel")
        label_field.setSFString(label)
        if visualize:
            self.supervisor.setLabel(0, label, 0.1, 0.5, 0.1, 0xFF0000, 0, "Impact")

    def update_training_target(self, label=""):
        """
        Updates the training task target.

        :param label: str, defaults to empty
        :return: None
        """
        label_field: Field = self.supervisor.getSelf().getField("trainingTaskTarget")
        label_field.setSFString(label)

    def use(self, target, destination: str, length=3, debug=True):
        """
        Performs the action "Use" (Place, Pick, Place)

        :param destination: str, name of the desired surface node
        :param length: number of place/pick cycles
        :return None
        """
        assert length > 0, "Lenght must be a positive integer"
        for i in range(length):
            # Tries to place, if successful tries to pick. If either one fails, stops the loop
            if debug:
                print("[DEBUG] Picking {0}".format(target))
            if self.pick(target):
                if debug:
                    print("[DEBUG] Placing {0} on {1}".format(target, destination))
                if self.place(destination):
                    continue
            print("Use action failed")
            return False
        return True

    def pick_and_place(self, target: str, destination: str, speed=2, debug=True):
        """
        Pick and place.

        :param target: string, name of the object to pick
        :param destination: string, name of the destination node
        :return None
        """
        self.busy_waiting(2, label="STILL")
        if debug:
            print("[DEBUG] Approaching: {0}".format(target))
        if self.approach_target(target, speed, debug):
            if debug:
                print("[DEBUG] Arrived at {0}. Attempting to pick.".format(target))
            self.pick(target)
            if debug:
                print("[DEBUG] Approaching: {0}".format(destination))
            self.approach_target(destination, speed, debug)
            if debug:
                print("[DEBUG] Arrived at {0}. Attempting to place.".format(destination))
            self.place(destination)
        return

    def relocate(self, coordinates):
        """
        Performs the action "Relocate" (Still, Walk, Still)

        :param coordinates: (X, Y) coordinates
        :return None
        """
        self.busy_waiting(3, label="STILL")
        self.approach_coordinates(coordinates, speed=2)
        self.busy_waiting(3, label="STILL")

    # GOALS #

    def breakfast(self, with_collab=False):
        """
        Enact the goal: "make breakfast".

        @param with_collab: if True, the human will expect the robot to collaborate
        @return: None
        """
        self.pick_and_place("biscuits", "plate")
        self.use("biscuits", "plate")
        if not with_collab:
            self.pick_and_place("plate", "sink")
            self.use("plate", "sink")
        else:
            self.relocate(tuple(self.startingPosition[:2]))    # Return to the original starting position

    def lunch(self, with_collab=False):
        """
        Enact the goal: "have lunch".

        @param with_collab: if True, the human will expect the robot to collaborate
        @return: None
        """
        self.pick_and_place("meal", "hobs")
        self.use("meal", "hobs")
        self.pick_and_place("meal", "plate")
        self.use("meal", "plate")
        if not with_collab:
            self.pick_and_place("plate", "sink")
            self.use("plate", "sink")
        else:
            self.relocate(tuple(self.startingPosition[:2]))    # Return to the original starting position

    def drink(self, with_collab=False):
        """
        Enact the goal: "drink some water".

        @param with_collab: if True, the human will expect the robot to collaborate
        @return: None
        """
        #self.pick_and_place("bottle", "worktop(4)")
        self.pick_and_place("bottle", "glass")
        self.use("glass", "bottle")
        if not with_collab:
            self.pick_and_place("glass", "sink")
            self.use("glass", "sink")
        else:
            self.relocate(tuple(self.startingPosition[:2]))    # Return to the original starting position

    def calculate_orientation_vector(self):
        """
        Calculates a point at distance = 1 from the human, taking into account its orientation in space.
        """
        human_position = self.get_robot_position()
        R = self.get_robot_orientation()
        T = np.array([human_position[0], human_position[1], 1.27])
        p_local = np.array([1, 0, 0], dtype=float)#[0, 1, 0]
        p_global = np.dot(R, p_local) + T
        new_position = [round(p_global[0], 2), round(p_global[1], 2)]
        dx = human_position[0] - new_position[0]
        dy = human_position[1] - new_position[1]
        return np.asarray([dx, dy]), human_position

    def angle_with(self, target_name: str, orientation_vector, human_position):
        """
        Calculates the angle between the human's orientation and a specified object.

        :param target_name: The object of interest.
        :param orientation_vector: The human's orientation vector, calculated previously.
        :return Angle, in radians
        """
        assert isinstance(target_name, str), "Must specify a string name to search for"
        target: Node = self.all_nodes.get(target_name)
        if target is not None:
            # Calculate the human-to-object vector
            target_trans = target.getPosition()
            object_position = np.asarray([target_trans[0], target_trans[1]])
            dx = human_position[0] - object_position[0]
            dy = human_position[1] - object_position[1]
            object_vector = np.asarray([dx, dy])
            unit_vector_1 = orientation_vector / np.linalg.norm(orientation_vector)
            unit_vector_2 = object_vector / np.linalg.norm(object_vector)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)
            angle_deg = round(math.degrees(angle))
            return angle_deg


# MAIN LOOP

def main():
    human = HumanAgent()
    while human.step():
        human.busy_waiting(1, label="STILL")    # Intro
        #human.breakfast(with_collab=True)
        human.lunch(with_collab=True)
        #human.drink(with_collab=True)
        human.busy_waiting(-1, label="STILL")   # Outro
        break
    print("HumanAgent: Simulation ended.")


# Run this code to benchmark execution time
#cProfile.run('main()', sort='time')
main()
