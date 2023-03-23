#!/usr/bin/env python
import message_filters
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
import rospy

def callback(ouster, rsu, tf):
    print("synchronization")

def tf_callback(msg):
    tf = TransformStamped()
    tf = msg.transforms[0]

def main():
    rospy.init_node('v2x_synchronizer_node')
    ouster_sub = message_filters.Subscriber('velodyne_points', PointCloud2)
    rsu_sub = message_filters.Subscriber('ZOE3/os_cloud_node/points', PointCloud2)
    tf_sub =  rospy.Subscriber('tf/transforms', tf_callback)

    ts = message_filters.TimeSynchronizer([ouster_sub, rsu_sub, tf_sub], 10)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    main()
