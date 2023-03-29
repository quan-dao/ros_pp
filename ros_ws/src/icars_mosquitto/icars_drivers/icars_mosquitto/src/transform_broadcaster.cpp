#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Pose.h>
#include "icars_mosquitto/Status.h"


std::string turtle_name;

void poseCallback(const icars_mosquitto::Status& msg){
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;
  
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = "map";
  transformStamped.child_frame_id = "ZOE3/os_sensor";

  transformStamped.transform.translation.x = msg.xGlobal;
  transformStamped.transform.translation.y = msg.yGlobal;
  transformStamped.transform.translation.z = 1.7;

  tf2::Quaternion q;
  q.setRPY(0.0,0.0, msg.heading);

  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();
  transformStamped.transform.rotation.w = q.w();

  br.sendTransform(transformStamped);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "my_tf2_broadcaster");

  ros::NodeHandle private_node("~");    
  ros::NodeHandle node;
  ros::Subscriber sub = node.subscribe("CAR/pose", 10, &poseCallback);
  while (ros::ok())
  {
  ros::spin();
  }

  return 0;
};