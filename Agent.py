"""
Generic Agent controller.
Groups common methods between Humans and Robots.
"""

from controller import Robot, Supervisor, Camera, RangeFinder, CameraRecognitionObject, GPS, DifferentialWheels
import numpy as np
import cv2

class Agent:
    def __init__(self, supervisor=False):
        # Create the instance (Robot or Supervisor)
        if supervisor:
            self.supervisor = Supervisor()
            self.robot = self.supervisor.getSelf()
        else:
            self.robot = Robot()

        # Get the time step of the current world
        if supervisor:
            self.timestep = int(self.supervisor.getBasicTimeStep())
        else:
            self.timestep = int(self.robot.getBasicTimeStep())

        # Initialize devices
        self.camera = Camera("camera")
        self.rangefinder = RangeFinder("range-finder")
        self.gps = GPS("gps")

        # Enable devices
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        #self.camera.enableRecognitionSegmentation()
        self.rangefinder.enable(self.timestep)
        self.gps.enable(self.timestep)

    def is_camera_active(self):
        """
        Checks whereas the camera has been enabled on the robot.

        :return: True or False
        """
        if self.camera.getSamplingPeriod() == 0:
            print("[ERROR] Camera on robot " + self.robot.getName() + " is not enabled.")
            return False
        return True

    def is_range_finder_active(self):
        """
        Checks whereas the range finder has been enabled on the robot.

        :return: True or False
        """
        if self.rangefinder.getSamplingPeriod() == 0:
            print("[ERROR] Range finder on robot " + self.robot.getName() + " is not enabled.")
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
                print(object_models)
            return objects

    def get_object_from_set(self, target, objects):
        """
        Searches for the presence of a specific object in a list of CameraRecognitionObject and returns it.

        :param target: str, (model) name of the desired object (e.g. 'can', 'pedestrian'...)
        :param objects: list of CameraRecognitionObject
        :return: CameraRecognitionObject corresponding to the desired target, or None if not found
        """
        object_models = [object.get_model().decode('utf-8') for object in objects]
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

    def get_gps_position(self):
        """
        Sugar code to fetch GPS coordinates and convert them to a 2D position on the XZ plane.

        :return: 2d world coordinates (x, z)
        """
        world_position = self.gps.getValues()
        return self.convert_to_2d_coords(world_position)

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
        if self.supervisor:
            entity = self.supervisor
        else:
            entity = self.robot
        return entity.step(self.timestep) != -1


# # MAIN LOOP
#
# agent = Agent(supervisor=True)
#
# # Perform simulation steps until Webots is stopping the controller
# while agent.step():
#     objects = agent.observe(True)
#     agent.show_camera_image(objects)