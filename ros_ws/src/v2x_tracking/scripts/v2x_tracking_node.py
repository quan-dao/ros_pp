#!/usr/bin/env python
# ROS-Python dependencies
import rospy
import numpy as np
import time 
import matplotlib.pyplot as plt
# ROS sensor_msg dependencies
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray
# from builtin_interfaces.msg import Duration
import sensor_msgs.point_cloud2 as pc2
# Custom Tracking library dependencies
from  tools.run_det_track import Detector, Tracktor


def euler_to_quaternion(yaw, pitch=0.0, roll=0.0):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return Quaternion(x=qx,y=qy,z=qz,w=qw)


def viz(boxes):
    msg_array = MarkerArray()
    # self.get_logger().info('Msgs received, time=%s' % time_stamp)*
    for box in boxes:
        msg = Marker()
        msg.type = 1
        msg.id = int(box[7])
        msg.header.frame_id = msg.header.frame_id
        msg.scale.x, msg.scale.y, msg.scale.z = box[3], box[4], box[5]
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = box[0], box[1], box[2]
        msg.pose.orientation = euler_to_quaternion(box[6])
        if box[-1] > 0.0:
            msg.color = ColorRGBA(r=1., a=0.5)
        else:
            msg.color = ColorRGBA(g=1., a=0.5)
        msg.lifetime = rospy.Duration(secs=0, nsecs=500000000)
        msg_array.markers.append(msg)

    return msg_array

def LIDAR_cb(msg):
    
    tic = time.time()
    
    # Convert PointCloud2 in numpy array of size (N, 6) - batch_idx, x, y, z, intensity, time_lag (== 0.0)
    points = np.asarray(list(pc2.read_points(msg, skip_nans=True)))[:,:6]
    # stamp_nsecs = msg.header.stamp.nsecs
    # stamp_secs = data.header.stamp.secs
    points[:,4] = 0.0
    points = np.pad(points, pad_width=[(0, 0), (1, 0)], constant_values=0.0)
    
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
    
    track_result = np.concatenate([track_ped, tracked_cars])
    pub.publish(viz(track_result))
    


def main():
    rospy.init_node('v2x_tracking_node')

    rospy.Subscriber("velodyne_points", PointCloud2, LIDAR_cb)
    # rospy.Subscriber("test", String, callback)
    # rospy.Subscriber("ZOE3/os_cloud_node/points", PointCloud2, callback)
    # points = np.array([[1,2,3,4,5,6]]) # dummy value
    
    # MarkerArray = np.array([1,2,3,4]) # dummy value
    rospy.spin()

if __name__ == '__main__':
    # Initialisation of ROS publisher
    pub = rospy.Publisher("test", MarkerArray, queue_size=10)
    # Initialisation of the detector and tracktor
    detector = Detector(score_threshold=0.2)
    tracktor_ped = Tracktor(chosen_class_index=8, cost_threshold=2.5, num_miss_to_kill=5)
    tracktor_car = Tracktor(chosen_class_index=0, cost_threshold=2.5, track_couters_init=10000, num_miss_to_kill=5)
    main()

	
