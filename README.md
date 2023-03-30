## Project roadmap

Gitlab for ROS usage: <https://gitlab.univ-nantes.fr/beaune-c/v2x_perception>

* [x] Transform PointCloud2 (ROS2 format) into Numpy tables
* [x] Use object detection and tracking algorithm on clouds
  * [x] Compute bounding boxes of objects for each timestamp
* [x] Send the poses of bounding boxes to vehicle
  * [x] Transform in vehicle's frame
  * [ ] Project it into camera image
* [ ] Build use cases of enhanced perception with communication

## Setup for the demo RSU side

1. Setup the velodyne on the RSU
   1. Connect it to the computer with following commands

      ```
      sudo ifconfig eno2 192.168.3.100 //need to verify
      sudo ifconfig route 192.168.100.1 eno2 //need to verify
      ```

      The connexion can be verified using the command 

      ```
      sudo wireshark
      ```
   2. Launch the velodyne driver

      ```
      roslaunch velodyne_pointcloud VLP16_points.launch
      ```
   3. Veify the visualization on RViz with topics */velodyne_points* and frame *velodyne*
2. Launch det&track node

   ```
    roslaunch v2x_tracking experiment.launch
   ```

   Do not forget to build and source the catkin workspace
3. Launch mosquitto publisher

   ```
   rosrun icars_mosquitto icars_mosquitto_publisher
   ```

## Setup for the demo CAR side

1. Launch sensors in ZOE

   ```
    roslaunch zoe3 icars_real_time.launch
   ```
2. Launch mosquitto subscriber

   ```
   rosrun icars_mosquitto icars_mosquitto_subscriber
   ```

   Do not forget to build and source the catkin workspace
3. Visualize with rviz

   ```
   rviz -d <path-to-ros_ws>/src/v2x_tracking/config/rviz.rviz
   ```

   Replace *<path-to-ros_ws>* with right path