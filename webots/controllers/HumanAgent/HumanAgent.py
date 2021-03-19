"""
HumanAgent controller.
Derives from Agent.
"""

from Agent import Agent
from controller import Node, Field, TouchSensor
import math
from controller import CameraRecognitionObject


class HumanAgent(Agent):
    def __init__(self, ):
        Agent.__init__(self, supervisor=True)

        self.all_nodes = self.obtain_all_nodes()
        self.object_in_hand = None

        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        self.speed = 1.15
        self.current_height_offset = 0
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

        print(str(self.__class__.__name__) + " has activated.")

    def get_robot_position(self):
        """
        Using Supervisor functions, obtains the 2D (x,z) position of the robot.

        :return: (x,z) position of the robot
        """
        return self.convert_to_2d_coords(self.supervisor.getSelf().getPosition())

    def hand_forward(self):
        """
        Adopts a position where the hand is leaning frontally.

        :return: None
        """
        # Target angles, defined empirically
        upper_arm_target = -0.485398
        lower_arm_target = -0.3
        upper_arm_index = self.joint_names.index("rightArmAngle")
        lower_arm_index = self.joint_names.index("rightLowerArmAngle")
        while self.step():
            self.joints_position_field[upper_arm_index].setSFFloat(upper_arm_target)
            self.joints_position_field[lower_arm_index].setSFFloat(lower_arm_target)
            self.step()
            break

    def walk(self, trajectory, speed=None, debug=False):
        """
        Executes a walking animation to a specified target coordinate.

        :param trajectory: trajectory in the format [(x1, y1), (x2, y2), ...]
        :param speed: walking speed in [m/s]
        :return:
        """
        if speed is None:
            speed = self.speed
        # Calculates the waypoints (including the starting position)
        initial_position = self.get_robot_position()
        number_of_waypoints = len(trajectory) + 1  # Because I add the initial waypoint by hand
        waypoints = [[initial_position[0], initial_position[1]]]
        for i in range(0, number_of_waypoints - 1):
            waypoints.append([trajectory[i][0], trajectory[i][1]])
        final_waypoint = waypoints[-1]

        print("I am about to walk from {0} to {1}".format(initial_position, final_waypoint))

        root_node_ref = self.supervisor.getSelf()   # root_node_ref is the Robot node where the Supervisor is running
        root_translation_field = root_node_ref.getField("translation")
        root_rotation_field = root_node_ref.getField("rotation")
        #for i in range(0, self.BODY_PARTS_NUMBER):
        #    joint = root_node_ref.getField(self.joint_names[i])
        #    self.joints_position_field.append(joint)

        # compute waypoints distance
        waypoints_distance = []
        for i in range(0, number_of_waypoints):
            x = waypoints[i][0] - waypoints[(i + 1) % number_of_waypoints][0]
            z = waypoints[i][1] - waypoints[(i + 1) % number_of_waypoints][1]
            if i == 0:
                waypoints_distance.append(math.sqrt(x * x + z * z))
            else:
                waypoints_distance.append(waypoints_distance[i - 1] + math.sqrt(x * x + z * z))
        while self.step():
            time = self.supervisor.getTime()

            current_sequence = int(((time * speed) / self.CYCLE_TO_DISTANCE_RATIO) % self.WALK_SEQUENCES_NUMBER)
            # compute the ratio 'distance already covered between way-point(X) and way-point(X+1)'
            # / 'total distance between way-point(X) and way-point(X+1)'
            ratio = (time * speed) / self.CYCLE_TO_DISTANCE_RATIO - \
                    int(((time * speed) / self.CYCLE_TO_DISTANCE_RATIO))

            for i in range(0, self.BODY_PARTS_NUMBER):
                current_angle = self.angles[i][current_sequence] * (1 - ratio) + \
                                self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                # Positions the joints
                self.joints_position_field[i].setSFFloat(current_angle)

            # adjust height
            self.current_height_offset = self.height_offsets[current_sequence] * (1 - ratio) + \
                                         self.height_offsets[
                                             (current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio

            # move everything forward
            distance = time * speed
            relative_distance = distance - int(distance / waypoints_distance[number_of_waypoints - 1]) * \
                                waypoints_distance[number_of_waypoints - 1]

            for i in range(0, number_of_waypoints):
                if waypoints_distance[i] > relative_distance:
                    break

            distance_ratio = 0
            if i == 0:
                distance_ratio = relative_distance / waypoints_distance[0]
            else:
                distance_ratio = (relative_distance - waypoints_distance[i - 1]) / \
                                 (waypoints_distance[i] - waypoints_distance[i - 1])
            x = distance_ratio * waypoints[(i + 1) % number_of_waypoints][0] + \
                (1 - distance_ratio) * waypoints[i][0]
            z = distance_ratio * waypoints[(i + 1) % number_of_waypoints][1] + \
                (1 - distance_ratio) * waypoints[i][1]
            root_translation = [x, self.ROOT_HEIGHT + self.current_height_offset, z]
            angle = math.atan2(waypoints[(i + 1) % number_of_waypoints][0] - waypoints[i][0],
                               waypoints[(i + 1) % number_of_waypoints][1] - waypoints[i][1])
            rotation = [0, 1, 0, angle]

            root_translation_field.setSFVec3f(root_translation)
            root_rotation_field.setSFRotation(rotation)
            self.move_object_in_hand()

            # Check if destination is reached
            agent_position = human.get_2d_position()
            dist = math.hypot(agent_position[0] - final_waypoint[0], agent_position[1] - final_waypoint[1])
            if debug:
                print("Current position: " + str(agent_position))
                print("Target position: " + str(final_waypoint))
                print("Distance: " + str(dist))
                print("---")
            # Stop when at 40 cm from target
            if dist <= 0.4:
                # Let's reset the position to a neutral pose
                # Ensure feet are touching the ground
                root_translation_field.setSFVec3f([x, self.ROOT_HEIGHT, z])
                # Adopt a neutral position
                for i in range(0, self.BODY_PARTS_NUMBER):
                    self.joints_position_field[i].setSFFloat(0.0)
                # Compute an additional step to ensure that the position is reached
                self.step()
                return

    def approach_target(self, target_name, debug=False):
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
            # my_position = self.get_2d_position()
            # Moves from the current location to the target position
            # if debug:
            #    print("Issued walk command from " + str(my_position) + " to " + str(target_position))
            # self.walk([my_position, target_position], debug=False)
            self.walk([target_position], debug=False)
            print("I have stopped at {0}".format(self.get_robot_position()))
            return True
        return False

    def grasp_object(self, target_name : str):
        """
        Positions the human in a grasping pose, then attaches the object (specified by its name) to the hand. Warning:
        the object is grasped even if it is not in range!

        :param target_name: str, name of the object to grasp
        :return: True if successful, False otherwise
        """
        assert isinstance(target_name, str), "Must specify a string name to search for"
        self.hand_forward()
        target: Node = self.all_nodes.get(target_name)
        if target is not None:
            self.object_in_hand = target
            self.move_object_in_hand()
            return True
        return False

    def release_object(self):
        if self.object_in_hand is not None:
            self.hand_forward()
            self.object_in_hand = None
            # todo => position the object on the surface?
            return True
        return False

    def move_object_in_hand(self):
        """
        Called to update the position of the held object to match the one of the hand.

        :return: None
        """
        if self.object_in_hand is not None:
            hand_translation, hand_rotation = self.get_hand_transform()
            self.object_in_hand.getField("translation").setSFVec3f(hand_translation)
            self.object_in_hand.getField("rotation").setSFRotation(hand_rotation)

    def get_hand_transform(self):
        """
        Obtains the translational and rotational vectors for the right hand.

        :return: hand translation, hand rotation wrt the world coordiantes
        """
        # Robot
        robot = self.supervisor.getSelf()
        robot_trans = robot.getField("translation").getSFVec3f()
        robot_rot = robot.getField("rotation").getSFRotation()
        robot_children = robot.getProtoField("children")
        robot_solid = robot_children.getMFNode(robot_children.getCount() - 1)
        robot_solid_children = robot_solid.getField("children")
        # Right upper arm
        upper_arm = robot_solid_children.getMFNode(3)
        upper_arm_endpoint = upper_arm.getField("endPoint")
        upper_arm_endpoint_solid = upper_arm_endpoint.getSFNode()
        upper_arm_trans = upper_arm_endpoint_solid.getField("translation").getSFVec3f()
        upper_arm_rot = upper_arm_endpoint_solid.getField("rotation").getSFRotation()
        upper_arm_children = upper_arm_endpoint_solid.getField("children")
        # Right lower arm
        hj = upper_arm_children.getMFNode(upper_arm_children.getCount() - 1)
        lower_arm_endpoint: Field = hj.getField("endPoint")
        lower_arm_endpoint_solid = lower_arm_endpoint.getSFNode()
        lower_arm_trans = lower_arm_endpoint_solid.getField("translation").getSFVec3f()
        lower_arm_rot = lower_arm_endpoint_solid.getField("rotation").getSFRotation()
        lower_arm_children = lower_arm_endpoint_solid.getField("children")
        # Hand
        hj2 = lower_arm_children.getMFNode(lower_arm_children.getCount() - 1)
        hand_endpoint: Field = hj2.getField("endPoint")
        hand_endpoint_solid = hand_endpoint.getSFNode()
        #hand_trans = hand_endpoint_solid.getField("translation").getSFVec3f()
        hand_children = hand_endpoint_solid.getField("children")
        # Hand_transform
        transform = hand_children.getMFNode(0)
        hand_trans = transform.getField("translation").getSFVec3f()
        hand_rot = transform.getField("rotation").getSFRotation()

        hand_position = transform.getPosition()
        #print("POS: " + str(transform.getPosition()))
        # Calculates the translation from the hand to the world
        translations = [robot_trans, upper_arm_trans, lower_arm_trans, hand_trans]
        rotations = [robot_rot, upper_arm_rot, lower_arm_rot, hand_rot] # disattivo?
        t = [sum(x) for x in zip(*translations)]
        r = [sum(x) for x in zip(*rotations)]
        #return t, r
        return hand_position, r

    def obtain_all_nodes(self):
        """
        Retrieves all the nodes from the current world and places them in a dictionary.

        :return: dictionary[name] = node
        """
        root_node = self.supervisor.getRoot()
        root_node_children: Field = root_node.getField("children")
        n = root_node_children.getCount()
        all_nodes = {}
        for i in range(n):
            node = root_node_children.getMFNode(i)
            name_field = node.getField("name")
            # We just care about objects, we discard non-named entities
            if name_field is not None:
                name = name_field.getSFString()
                all_nodes[name] = node
        return all_nodes


# MAIN LOOP

human = HumanAgent()

while human.step():
    human.approach_target("coca-cola", True)
    #human.hand_forward()
    human.grasp_object("coca-cola")
    human.approach_target("Peppe", True)
    break
