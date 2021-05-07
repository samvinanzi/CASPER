"""
Generic Agent controller.
Groups common methods between Humans and Robots.
"""

from controller import Robot, Supervisor, Camera, RangeFinder, CameraRecognitionObject, GPS, TouchSensor, \
    DistanceSensor, Field
import numpy as np
import cv2


class Agent:
    def __init__(self, debug=False):
        self.supervisor = Supervisor()                              # All agents have access to Supervisor functions
        self.timestep = int(self.supervisor.getBasicTimeStep())     # Get the time step of the current world
        self.currentlyPlaying = None                                # Currently playing motion
        self.all_nodes = self.obtain_all_nodes()        # Contains all the nodes in the scene, for Supervisory purposes
        self.debug = debug

        # Initialize common devices
        self.camera = Camera("camera")
        self.rangefinder = RangeFinder("range-finder")
        #self.gps = GPS("gps")
        #self.bumper = TouchSensor("touch sensor")
        #self.distance_sensor = DistanceSensor("distance sensor")

    def obtain_all_nodes(self):
        """
        Retrieves all the nodes from the current world and places them in a dictionary.

        :return: dictionary[name] = node
        """
        root_node = self.supervisor.getRoot()
        if root_node is not None:
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
        else:
            return None

    def is_camera_active(self):
        """
        Checks whereas the camera has been enabled on the robot.

        :return: True or False
        """
        if self.camera.getSamplingPeriod() == 0:
            print("[ERROR] Camera on robot " + self.supervisor.getName() + " is not enabled.")
            return False
        return True

    def is_range_finder_active(self):
        """
        Checks whereas the range finder has been enabled on the robot.

        :return: True or False
        """
        if self.rangefinder.getSamplingPeriod() == 0:
            print("[ERROR] Range finder on robot " + self.supervisor.getName() + " is not enabled.")
            return False
        return True

    def observe(self, debug=False):
        """
        Observes the scene to identify objects.

        :param debug: if True, prints a verbose description of the observed scene
        :return: CameraRecognitionObject list
        """
        if self.is_camera_active():
            n_obj = self.camera.getRecognitionNumberOfObjects()
            objects = self.camera.getRecognitionObjects()
            if debug:
                print("I have recognized " + str(n_obj) + " object" + ("s" if n_obj != 1 else "") + "!")
                print("Objects detected: ")
                object_models = [object.get_model().decode('utf-8') for object in objects]
            return objects

    def objects_models(self, objects):
        """
        Converts a list of CameraRecognitionObject into a list of strings with their models.

        :param objects: CameraRecognitionObject list
        :return: str list
        """
        return [object.get_model().decode('utf-8') for object in objects]

    def get_object_from_set(self, target, objects):
        """
        Searches for the presence of a specific object in a list of CameraRecognitionObject and returns it.

        :param target: str, (model) name of the desired object (e.g. 'can', 'pedestrian'...)
        :param objects: list of CameraRecognitionObject
        :return: CameraRecognitionObject corresponding to the desired target, or None if not found
        """
        object_models = self.objects_models(objects)
        try:
            index = object_models.index(target)
            return objects[index]
        except ValueError:
            return None

    def get_object_distance(self, object: CameraRecognitionObject):
        """
        Finds the depth value of an object recognized by the camera.

        :param object: CameraRecognitionObject item
        :return: Distance in meters
        """
        if self.is_range_finder_active():
            # Retrieve the coordinates of the object
            coordinates = tuple(object.get_position_on_image())
            depth_image = self.rangefinder.getRangeImage()
            # Gets the depth value from the depth image on the object coordinates
            depth = self.rangefinder.rangeImageGetDepth(depth_image, self.rangefinder.getWidth())
            return depth

    def convert_to_2d_coords(self, world_coordinates):
        """
        Converts a 3D world coordinate in a 2D coordinate (for i.e. trajectories).

        :param world_coordinates: 3d world coordinates (x, y, z)
        :return: 2d world coordinates (x, z)
        """
        return (round(world_coordinates[0], 2), round(world_coordinates[2], 2))

    def is_bumper_pressed(self):
        """
        Checks wherever the bumper has detected a force (collision) or not.

        :return: True if a contact has happened, False otherwise
        """
        return bool(self.bumper.getValue())

    def show_camera_image(self, objectlist=None, segmented=False):
        """
        Displays the camera stream from the robot on an OpenCV window.

        :param objectlist: CameraRecognitionObject list to mark on screen
        :param segmented: shows the segmented recognized objects instead of the full view
        :return: None
        """
        if self.is_camera_active():
            if not segmented:
                camera_data = self.camera.getImage()
            else:
                camera_data = self.camera.getRecognitionSegmentationImage()
            image = np.frombuffer(camera_data, np.uint8).reshape(
                (self.camera.getHeight(), self.camera.getWidth(), 4))
            if objectlist is not None:
                for object in objectlist:
                    if isinstance(object, CameraRecognitionObject):
                        coordinates = tuple(object.get_position_on_image())
                        shifted_coordinates = (coordinates[0], coordinates[1] - 10)
                        model = object.get_model().decode('utf-8')
                        size = object.get_size_on_image()
                        color = (0, 0, 255)
                        # Draw on the image
                        cv2.circle(image, coordinates, 3, color, thickness=-1)
                        cv2.putText(image, model, shifted_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            cv2.imshow("Robot view", image)
            cv2.waitKey(self.timestep)

    def save_camera_image(self, filename="camera_frame", format="jpg", quality=100, segmented=False):
        """
        Saves the current camera image.

        :param filename: Name of the file
        :param format: PNG, JPG, JPEG or HDR
        :param quality: 1-100, 100 being the best
        :param segmented: If True, saves the segmented objects recognized
        :return: None
        """
        assert format.lower() in ["png", "jpg", "jpeg", "hdr"], "Invalid image format requested."
        assert 1 <= quality <= 100, "Quality parameters must be within boundaries [1, 100]."
        if self.is_camera_active():
            if not segmented:
                self.camera.saveImage(str(filename + "." + format), quality)
            else:
                self.camera.saveRecognitionSegmentationImage(str(filename + "." + format), quality)

    def save_rangefinder_image(self, filename="rangefinder_frame", format="jpg", quality=100):
        """
        Saves the current range finder image.

        :param filename: Name of the file
        :param format: PNG, JPG, JPEG or HDR
        :param quality: 1-100, 100 being the best
        :return: None
        """
        assert format.lower() in ["png", "jpg", "jpeg", "hdr"], "Invalid image format requested."
        assert 1 <= quality <= 100, "Quality parameters must be within boundaries [1, 100]."
        if self.is_camera_active():
            self.rangefinder.saveImage(str(filename + "." + format), quality)

    def step(self):
        """
        Sugar code wrapper for the Webots step function.

        :return: step
        """
        return self.supervisor.step(self.timestep) != -1
