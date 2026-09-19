#!/usr/bin/env python3


'''
*****************************************************************************************
*
*               ===============================================
*                           StrataCobot (SC) Theme (eYRC 2026-27)
*               ===============================================
*
*  This script should be used to implement Task 1A of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          3809
# Author List:      Vishwa G S, Pranesh S, Dharrenya S A, Pranesh G A 
# Filename:         task1A.py
# Functions:        detect_ores, depthimagecb, colorimagecb, caminfocb, process_image, main
# Nodes:            ore_tf_publisher
#                   Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw,
#                                          /camera/camera/aligned_depth_to_color/image_raw,
#                                          /camera/camera/color/camera_info ]

################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point


##################### TASK CONSTANTS #######################

# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# The RealSense topics. The depth image is ALIGNED to the colour image.
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

# The parent frame every ore transform is published against.
base_frame = 'base_link'

# Ore geometry
# The camera detects the top face, but TF must represent the ore centre.
ORE_HALF_HEIGHT = 0.0381  # meters


##################### FUNCTION DEFINITIONS #######################

def detect_ores(image):

    center_ore_list = []
    ore_type_list = []

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # =========================================================
    # AZURITE - BLUE
    # =========================================================

    lower_blue = np.array([95, 150, 80])
    upper_blue = np.array([115, 255, 255])

    # Create blue mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find blue contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Process contours
    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small regions
        if area < 100:
            continue

        # Calculate centre
        M = cv2.moments(contour)

        if M['m00'] == 0:
            continue

        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # Store detection
        center_ore_list.append((cX, cY))
        ore_type_list.append('azurite_ore')

        # Draw contour
        cv2.drawContours(
            image,
            [contour],
            -1,
            (255, 255, 255),
            2
        )

        # Draw centre
        cv2.circle(
            image,
            (cX, cY),
            5,
            (255, 255, 255),
            -1
        )

    # =========================================================
    # MALACHITE - GREEN
    # =========================================================

    lower_green = np.array([40, 120, 60])
    upper_green = np.array([85, 255, 255])

    mask_green = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    mask_green = cv2.morphologyEx(
        mask_green,
        cv2.MORPH_OPEN,
        kernel
    )

    mask_green = cv2.morphologyEx(
        mask_green,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours_green, _ = cv2.findContours(
        mask_green,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours_green:

        area = cv2.contourArea(contour)

        if area < 100:
            continue

        M = cv2.moments(contour)

        if M['m00'] == 0:
            continue

        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        center_ore_list.append((cX, cY))
        ore_type_list.append('malachite_ore')

        cv2.drawContours(
            image,
            [contour],
            -1,
            (255, 255, 255),
            2
        )

        cv2.circle(
            image,
            (cX, cY),
            5,
            (255, 255, 255),
            -1
        )


        # =========================================================
    # VANADINITE - ORANGE
    # =========================================================

    lower_orange = np.array([5, 200, 100])
    upper_orange = np.array([18, 255, 255])

    # Create orange mask
    mask_orange = cv2.inRange(
        hsv,
        lower_orange,
        upper_orange
    )

    # Remove small noise
    mask_orange = cv2.morphologyEx(
        mask_orange,
        cv2.MORPH_OPEN,
        kernel
    )

    # Close small gaps
    mask_orange = cv2.morphologyEx(
        mask_orange,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find orange contours
    contours_orange, _ = cv2.findContours(
        mask_orange,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Process contours
    for contour in contours_orange:

        area = cv2.contourArea(contour)

        # Ignore very small regions
        if area < 100:
            continue

        # Calculate centre
        M = cv2.moments(contour)

        if M['m00'] == 0:
            continue

        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # Store detection
        center_ore_list.append((cX, cY))
        ore_type_list.append('vanadinite_ore')

        # Draw boundary
        cv2.drawContours(
            image,
            [contour],
            -1,
            (255, 255, 255),
            2
        )

        # Draw centre
        cv2.circle(
            image,
            (cX, cY),
            5,
            (255, 255, 255),
            -1
        )
    return center_ore_list, ore_type_list


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to detect the ores in the cell and
                    broadcast a transform for each one.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ore_tf
        '''

        super().__init__('ore_tf_publisher')                                            # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)

        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())
        self.cam_info = None
        self.color_frame = None                                                         # camera frame_id variable (from colorimagecb())

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        #   ->  Add any variable your detection needs to keep between frames.
        #       ->  HINT: The two ores of a type must keep their ids for the whole run, and
        #                 'detect_ores' returns them unordered.

        ############################################


    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
                        Use this function to receive the depth image and convert it to a CV2 image.

        Args:
            data (Image):    Input depth image frame received from the aligned depth camera topic

        Returns:
        '''

        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                data,
                desired_encoding='passthrough'
            )
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')


    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw topic.
                        Use this function to receive the raw image and convert it to a CV2 image.

        Args:
            data (Image):    Input coloured raw image frame received from the image_raw camera topic

        Returns:
        '''

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(
                data, desired_encoding='bgr8'
            )

            self.color_frame = data.header.frame_id

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic.
                        Use this function to receive the camera's intrinsic parameters.

        Args:
            data (CameraInfo):    Camera calibration published by the camera

        Returns:
        '''

        self.fx = data.k[0]
        self.fy = data.k[4]
        self.cx = data.k[2]
        self.cy = data.k[5]

        self.cam_info = data


    def process_image(self):
        '''
        Description:    Timer function used to detect the ores and publish a transform for
                        each one on its estimated position.

        Args:
        Returns:
        '''

        if self.cv_image is None:
            return

        if self.depth_image is None:
            return

        if self.cam_info is None:
            return

        center_ore_list, ore_type_list = detect_ores(self.cv_image)

        # Store all successfully processed ores before assigning IDs.
        detected_ores = []

        for center, ore_type in zip(center_ore_list, ore_type_list):

            cX, cY = center

            # Get a 9x9 depth patch around the ore centre
            patch = self.depth_image[cY-4:cY+5, cX-4:cX+5]

            # Keep only valid depth values
            valid_depth = patch[
                np.isfinite(patch) & (patch > 0.0)
            ]

            # Skip detection if no valid depth is available
            if valid_depth.size == 0:
                continue

            # Median depth
            z = np.median(valid_depth)

            # Convert pixel coordinates to 3D coordinates
            x = (cX - self.cx) * z / self.fx
            y = (cY - self.cy) * z / self.fy

            # Create 3D point in camera optical frame
            point_in_camera = PointStamped()

            point_in_camera.header.frame_id = self.color_frame
            point_in_camera.header.stamp = self.get_clock().now().to_msg()

            point_in_camera.point.x = float(x)
            point_in_camera.point.y = float(y)
            point_in_camera.point.z = float(z)

            # Transform camera point into base_link
            try:
                tf = self.tf_buffer.lookup_transform(
                    base_frame,
                    self.color_frame,
                    rclpy.time.Time()
                )

                point_in_base = do_transform_point(
                    point_in_camera,
                    tf
                )

            except Exception as e:
                self.get_logger().warn(
                    f'TF lookup failed: {e}'
                )
                continue

            # Get transformed coordinates
            # The detected point lies on the top face of the ore.
            bx = point_in_base.point.x
            by = point_in_base.point.y
            bz_top = point_in_base.point.z

            # Move from the top face to the geometric centre of the ore.
            bz = bz_top - ORE_HALF_HEIGHT

            # Store the detected ore for ID assignment.
            detected_ores.append({
                'type': ore_type,
                'x': float(bx),
                'y': float(by),
                'z': float(bz),
                'pixel': (cX, cY)
            })

        # Group detections by ore type.
        grouped_ores = {
            'azurite_ore': [],
            'malachite_ore': [],
            'vanadinite_ore': []
        }

        for ore in detected_ores:
            grouped_ores[ore['type']].append(ore)

        # Assign stable IDs using base_link X-coordinate.
        named_ores = []

        for ore_type in ore_types:

            ores = grouped_ores[ore_type]

            ores_sorted = sorted(
                ores,
                key=lambda ore: (ore['x'], ore['y'])
            )

            for index, ore in enumerate(ores_sorted, start=1):

                ore['name'] = f'{ore_type}_{index}'

                named_ores.append(ore)

                self.get_logger().info(
                    f"{ore['name']}: "
                    f"pixel={ore['pixel']} "
                    f"base_link=({ore['x']:.3f}, "
                    f"{ore['y']:.3f}, "
                    f"{ore['z']:.3f})"
                )

        # =========================================================
        # PUBLISH TF - ALL SIX ORES
        # =========================================================

        for ore in named_ores:

            if ore['name'] not in [
                'azurite_ore_1',
                'azurite_ore_2',
                'malachite_ore_1',
                'malachite_ore_2',
                'vanadinite_ore_1',
                'vanadinite_ore_2'
            ]:
                continue

            transform = TransformStamped()

            # Parent frame
            transform.header.frame_id = base_frame

            # Child frame
            transform.child_frame_id = ore['name']

            # Timestamp
            transform.header.stamp = self.get_clock().now().to_msg()

            # Ore centre position in base_link
            transform.transform.translation.x = ore['x']
            transform.transform.translation.y = ore['y']
            transform.transform.translation.z = ore['z']

            # Identity quaternion
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0

            # Publish transform
            self.br.sendTransform(transform)

            # =========================================================
            # DRAW FINAL ORE NAME
            # =========================================================

            cX, cY = ore['pixel']

            cv2.putText(
                self.cv_image,
                ore['name'],
                (cX - 45, cY - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow('Task 1A - Ore Detection', self.cv_image)
        cv2.waitKey(1)


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the ore_tf
                    class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('ore_tf_process')                      # creating ROS node

    node.get_logger().info('Node created: Ore tf process')          # logging information

    ore_tf_class = ore_tf()                                         # creating a new object for class 'ore_tf'

    rclpy.spin(ore_tf_class)                                        # spining on the object to make it alive in ROS 2 DDS

    ore_tf_class.destroy_node()                                     # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':

    main()