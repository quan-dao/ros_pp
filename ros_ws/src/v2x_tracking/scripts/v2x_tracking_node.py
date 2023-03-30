#!/usr/bin/env python
# ROS-Python dependencies
import rospy
import numpy as np
import time 
import matplotlib.pyplot as plt
# ROS sensor_msg dependencies
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
# from builtin_interfaces.msg import Duration
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage
import tf2_ros
from tf.transformations import quaternion_from_euler
# Custom Tracking library dependencies
from tools.run_det_track import Detector, Tracktor
from functools import partial
from typing import List


def pub_list_obstacle(bboxes,frame_id):
    bboxes = bboxes.reshape(-1).tolist()
    array = Float32MultiArray()
    array.data = bboxes
    obstacle_publisher.publish(array)


def euler_to_quaternion(yaw, pitch=0.0, roll=0.0):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return Quaternion(x=qx,y=qy,z=qz,w=qw)


def viz(boxes, frame):
    msg_array = MarkerArray()
    # self.get_logger().info('Msgs received, time=%s' % time_stamp)*
    occlusion_array = compute_occlusion_flag(boxes)
    for idx, box in enumerate(boxes):
        msg = Marker()
        msg.type = 1
        msg.ns = "obstacle"
        msg.id = int(box[7])
        msg.header.frame_id = frame
        msg.scale.x, msg.scale.y, msg.scale.z = box[3], box[4], box[5]
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = box[0], box[1], box[2]
        msg.pose.orientation = euler_to_quaternion(box[6])

        msg_text = Marker()
        msg_text.type = 9
        msg_text.ns = "text"
        msg_text.id = int(box[7])
        msg_text.header.frame_id = frame
        msg_text.scale.x, msg_text.scale.y, msg_text.scale.z = 1.0, 1.0, 1.0
        msg_text.pose.position.x, msg_text.pose.position.y, msg_text.pose.position.z = box[0], box[1], box[2]
        msg_text.pose.orientation = euler_to_quaternion(box[6])
        msg_text.text = str(int(box[7]))
        msg_text.color = ColorRGBA(r=1., a=1.0)
        msg_text.lifetime = rospy.Duration(secs=0.2, nsecs=0.0)

        if occlusion_array[idx] == 1:
            msg.color = ColorRGBA(r=1., a=0.5)
        else:
            msg.color = ColorRGBA(g=1., a=0.5)
        msg.lifetime = rospy.Duration(secs=0.2, nsecs=0.0)
        msg_array.markers.append(msg)
        msg_array.markers.append(msg_text)

    return msg_array

def LIDAR_cb(msg, points_yaw_threshold_degree: float = None, points_depth_threshold: float = None):
    node_name = rospy.get_param("~node_name", "rsu")
    tf_ouster_to_map_transl = [
        rospy.get_param('~tf_ouster_to_map_transl_x', 0.),
        rospy.get_param('~tf_ouster_to_map_transl_y', 0.),
        rospy.get_param('~tf_ouster_to_map_transl_z', 0.)
    ]

    print(f'timestamp_{rospy.get_param("~node_name", "rsu")} {msg.header.stamp.secs}')
    tf_msg = TransformStamped()
    tf_msg.header.stamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
    tf_msg.header.frame_id = "map"
    tf_msg.child_frame_id = rospy.get_param("~frame_id", "ZOE3/os_sensor")
    print(f'{tf_msg.child_frame_id}')
    
    tf_ouster_to_map_yaw_rad = rospy.get_param("~tf_ouster_to_map_yaw_rad", 2.0176)
    q = quaternion_from_euler(0., 0., tf_ouster_to_map_yaw_rad)
    tf_msg.transform.translation.x = rospy.get_param('~tf_ouster_to_map_transl_x', 1186.51)
    tf_msg.transform.translation.y = rospy.get_param('~tf_ouster_to_map_transl_y', 756.045)
    tf_msg.transform.translation.z = rospy.get_param('~tf_ouster_to_map_transl_z', -0.00143)


    print(f'{tf_ouster_to_map_transl, tf_ouster_to_map_yaw_rad}')
    print('-----------------------------------')
    
    
    tf_msg.transform.rotation.x = q[0]
    tf_msg.transform.rotation.y = q[1]
    tf_msg.transform.rotation.z = q[2]
    tf_msg.transform.rotation.w = q[3]

    _msg = TFMessage([tf_msg])
    tf_publisher.publish(_msg)

    if not detect_and_track:    
        # early return
        dummy_bbox = np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                              [1.2,1.2,1.2,1.2,1.2,1.2,1.2]])
        pub_list_obstacle(dummy_bbox,"base_link")
        
        return
        
    # Convert PointCloud2 in numpy array of size (N, 6) - batch_idx, x, y, z, intensity, time_lag (== 0.0)
    points = np.asarray(list(pc2.read_points(msg, skip_nans=True)))[:,:5]
    # stamp_nsecs = msg.header.stamp.nsecs
    # stamp_secs = data.header.stamp.secs
    points[:,4] = 0.0

    if points_yaw_threshold_degree is not None:
        points_yaw_threshold = np.deg2rad(points_yaw_threshold_degree)
        assert points_yaw_threshold > 0, f"expect positive yaw threshold, get {points_yaw_threshold}"
        # remove points behind LiDAR
        points = points[points[:, 0] < -1]
        # remove points having outside of range
        yaw = np.arctan2(points[:, 0], points[:, 1])
        mask_yaw_in_range = np.abs(yaw) < points_yaw_threshold
        points = points[mask_yaw_in_range]

    if points_depth_threshold is not None:
        points_depth_sqr = np.square(points[:, :2]).sum(axis=1)
        mask_depth_in_range = points_depth_sqr < points_depth_threshold**2
        points = points[mask_depth_in_range]

    # pad points with batch_idx (==0 for online inference)
    points = np.pad(points, pad_width=[(0, 0), (1, 0)], constant_values=0.0)
    rospy.loginfo(f'points shape is {points.shape}')
    # detection
    boxes = detector.detect(points)
    # tracking ped
    tracktor_ped.update_(boxes)
    tracked_boxes, tracked_id = tracktor_ped.report_tracks()
    track_ped = np.concatenate([tracked_boxes, tracked_id.reshape(-1, 1)], axis=1)
    track_ped = np.pad(track_ped, pad_width=[(0, 0), (0, 1)], constant_values=8.)

    # tracking cars
    tracktor_car.update_(boxes)
    car_boxes, cars_id = tracktor_car.report_tracks()
    tracked_cars = np.concatenate([car_boxes, cars_id[:, np.newaxis]], axis=1)
    tracked_cars = np.pad(tracked_cars, pad_width=[(0, 0), (0, 1)], constant_values=0.)
    
    frame = msg.header.frame_id
    track_result = np.concatenate([track_ped, tracked_cars])
    pub.publish(viz(track_result, frame))

    pub_list_obstacle(track_result, frame)
    pub_list_obstacle(track_result,frame)


def compute_occlusion_flag(boxes: np.ndarray, yaw_range_iou_threshold: float = 0.5) -> np.ndarray:
    """
    Given a set of boxes in the local coordinate of a LiDAR, find out which ones are occluded

    Args:
        boxes: (N, 7 [+ C]) - x, y, z, dx, dy, dz, yaw, [others]
    
    Returns:
        occlusion_flag (N,): 1 -> occluded, 0 - not occluded
    """
    
    # sort boxes according to distance from their centers to LiDAR
    dist = np.square(boxes[:, :2]).sum(axis=1)
    sorted_idx = np.argsort(dist)
    boxes = boxes[sorted_idx]
    
    # -----------------------------
    # for each box get its yaw range
    # -----------------------------
    xs = np.array([1, -1, -1, 1]) / 2.0
    ys = np.array([1, 1, -1, -1]) / 2.0
    zs = np.array([1, 1, 1, 1]) / 2.0
    vers = np.stack([xs, ys, zs], axis=1)  # (4, 3) - normalized coordinate of boxes' vertices 
    boxes_vertices = boxes[:, np.newaxis, 3: 6] * vers[np.newaxis]  # (N, 4, 3) - xyz in boxes' local frame
    # ---
    # map boxes_vertices from local coord to LiDAR coord
    # ---
    cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    zeros, ones = np.zeros(boxes.shape[0]), np.ones(boxes.shape[0])
    lidar_rot_box_transpose = np.stack([
        cos, sin, zeros,
        -sin, cos, zeros,
        zeros, zeros, ones
    ], axis=1).reshape(boxes.shape[0], 3, 3)
    boxes_vertices = np.einsum('nij, njk -> nik', boxes_vertices, lidar_rot_box_transpose) \
        + boxes[:, np.newaxis, :3]  # (N, 4, 3) - xyz in LiDAR frame
    boxes_yaw_range = np.arctan2(boxes_vertices[:, :, 1], boxes_vertices[:, :, 0])  # (N, 4)
    boxes_yaw_range = np.stack([np.amin(boxes_yaw_range, axis=1), 
                                np.amax(boxes_yaw_range, axis=1)], axis=1)  # (N, 2) - yaw_min, yaw_max
    
    # --------------------------------
    # compute IoU of boxes_yaw_range
    # --------------------------------
    boxes_yaw_inter_min = np.maximum(boxes_yaw_range[:, np.newaxis, 0], boxes_yaw_range[np.newaxis, :, 0])  # (N, N)
    boxes_yaw_inter_max = np.minimum(boxes_yaw_range[:, np.newaxis, 1], boxes_yaw_range[np.newaxis, :, 1])  # (N, N)
    boxes_yaw_inter = np.clip(boxes_yaw_inter_max - boxes_yaw_inter_min, a_min=0.0, a_max=None)  # (N, N)

    boxes_yaw = boxes_yaw_range[:, 1] - boxes_yaw_range[:, 0]  # (N,)
    boxes_yaw_union = boxes_yaw[:, np.newaxis] + boxes_yaw[np.newaxis, :]  # (N, N)

    boxes_yaw_iou = boxes_yaw_inter / boxes_yaw_union  # (N, N)
    boxes_yaw_iou = boxes_yaw_iou > yaw_range_iou_threshold

    # greedy occlusion marking -> like NMS
    occlusion_flag = np.zeros(boxes.shape[0], dtype=bool)
    for b_idx in range(boxes.shape[0]):
        boxes_yaw_iou[b_idx, b_idx] = 0.0  # mark a box's IoU with itself = 0.0 -> a box doesn't occlude itself
        occlusion_flag = np.logical_or(occlusion_flag, boxes_yaw_iou[b_idx])
    return occlusion_flag.astype(int)


def main():
    points_yaw_threshold_degree = rospy.get_param("~points_yaw_threshold_degree", None)
    points_yaw_threshold_degree = rospy.get_param("~points_depth_threshold", None)


    wrapper_lidar_cb = partial(LIDAR_cb, 
                               points_yaw_threshold_degree=points_yaw_threshold_degree, 
                               points_depth_threshold=points_yaw_threshold_degree)
    
    rospy.Subscriber("velodyne_points", PointCloud2, wrapper_lidar_cb)
    
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('v2x_tracking_node')
    # Initialisation of ROS publisher
    pub = rospy.Publisher("test", MarkerArray, queue_size=10)
    tf_publisher = rospy.Publisher("/tf", TFMessage, queue_size=1)
    obstacle_publisher = rospy.Publisher("obstacle_list", Float32MultiArray, queue_size=1)
    # Initialisation of the detector and tracktor
    detection_score_threshold = rospy.get_param("~detection_score_threshold", 0.2)
    tracking_cost_threshold_ped = rospy.get_param("~tracking_cost_threshold_ped", 2.5)
    tracking_cost_threshold_car = rospy.get_param("~tracking_cost_threshold_car", 5.5)
    num_miss_to_kill = rospy.get_param("~num_miss_to_kill", 10)
    
    detect_and_track = rospy.get_param("~detect_and_track", False)
    if detect_and_track:
        detector = Detector(score_threshold=detection_score_threshold)
        
        tracktor_ped = Tracktor(chosen_class_index=8, 
                                cost_threshold=tracking_cost_threshold_ped, 
                                num_miss_to_kill=num_miss_to_kill)
        
        tracktor_car = Tracktor(chosen_class_index=0, 
                                cost_threshold=tracking_cost_threshold_car, 
                                track_couters_init=10000, 
                                num_miss_to_kill=num_miss_to_kill)
    else:
        detector, tracktor_ped, tracktor_car = None, None, None
    main()

	
