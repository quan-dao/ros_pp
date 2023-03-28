#include <stdio.h>
#include <string.h>
#include <mosquitto.h>
#include <iostream>

#include "ros/ros.h"


#include "visualization_msgs/MarkerArray.h"
#include "visualization_msgs/Marker.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>

/*
 * This example shows how to write a client that subscribes to a topic and does
 * not do anything other than handle the messages that are received.
 */

#include <mosquitto.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using namespace std;
visualization_msgs::MarkerArray obstacles;


class MarkerArrayModifier
{
public:
  MarkerArrayModifier(string &topic, string &msg, visualization_msgs::MarkerArray &obstacles)
  {
	
	
	int id = topic_hl(topic);
	msg_hl(id, msg, obstacles);
	// for (auto &obstacle : obstacles.markers){
	// 	geometry_msgs::PoseStamped pose_in;
	// 	pose_in.header = obstacle.header;
	// 	pose_in.pose = obstacle.pose;
		
	// 	// tf::poseStampedMsgToTF(geometry_msgs::PoseStamped(obstacle.header, obstacle.pose), pose_in);
	// 	geometry_msgs::PoseStamped pose_out;
	// 	listener.transformPose("ZOE3/os_sensor", pose_in, pose_out);
	// 	obstacle.pose = pose_out.pose;
	// 	obstacle.header = pose_out.header;
	// 	cout <<pose_in << "   :  \n "<< pose_out<< endl;
	// 	cout << "-----------------------"<< endl;
	// }
	


        // Advertise the topic that provides the modified message
  }

private:
  ros::Subscriber sub_;
  ros::Publisher pub_;
  tf::TransformListener listener;
  visualization_msgs::MarkerArray modifiedMarkerArray_;

  int topic_hl(string &topic){
	stringstream ss(topic);
  	vector<string> tokens;

  	while (ss.good()) {
		string token;
		getline(ss, token, '/');
		tokens.push_back(token);
  }
//   cout<<stoi(tokens[1])<<endl;
  return stoi(tokens[1]);
  }
  
  
  void msg_hl(int &id, string &msg, visualization_msgs::MarkerArray &obstacles){
	stringstream ss(msg);
  	vector<string> tokens;

  	while (ss.good()) {
		string token;
		getline(ss, token, '/');
		tokens.push_back(token);
  }
//   listener.waitForTransform("velodyne", "ZOE3/os_sensor", ros::Time(0), ros::Duration(3.0));
	for(auto obstacle : obstacles.markers){
		if(id == obstacle.id){
			obstacle.pose.position.x = stof(tokens[0]);
			obstacle.pose.position.y = stof(tokens[1]);
			obstacle.pose.position.z =stof(tokens[2]);
			obstacle.scale.x = stof(tokens[3]);
			obstacle.scale.y = stof(tokens[4]);
			obstacle.scale.z = stof(tokens[5]);
			tf2::Quaternion q_heading;
			q_heading.setRPY( 0, 0, stof(tokens[6]));
			obstacle.pose.orientation.x = q_heading.getX();
			obstacle.pose.orientation.y = q_heading.getY();
			obstacle.pose.orientation.z = q_heading.getZ();
			obstacle.pose.orientation.w = q_heading.getW();

			// geometry_msgs::PoseStamped pose_in;
			// pose_in.header = obstacle.header;
			// pose_in.pose = obstacle.pose;
			// geometry_msgs::PoseStamped pose_out;
			// listener.transformPose("ZOE3/os_sensor", pose_in, pose_out);

			// obstacle.pose = pose_out.pose;
			// obstacle.header = pose_out.header;
					break;
		}
	
	}
	visualization_msgs::Marker new_obstacle;
	new_obstacle.ns = "obstacle";
	

	new_obstacle.pose.position.x = stof(tokens[0]);
	new_obstacle.pose.position.y = stof(tokens[1]);
	new_obstacle.pose.position.z = stof(tokens[2]);
	new_obstacle.scale.x = stof(tokens[3]);
	new_obstacle.scale.y = stof(tokens[4]);
	new_obstacle.scale.z = stof(tokens[5]);
	tf2::Quaternion q_heading;
	q_heading.setRPY( 0, 0, stof(tokens[6]));
	new_obstacle.pose.orientation.x = q_heading.getX();
	new_obstacle.pose.orientation.y = q_heading.getY();
	new_obstacle.pose.orientation.z = q_heading.getZ();
	new_obstacle.pose.orientation.w = q_heading.getW();
  
	new_obstacle.header.frame_id = "velodyne";
	new_obstacle.type = 1;
	if (stoi(tokens[7]) == 0){
		new_obstacle.color.g = 0.5;
		}
	else  {new_obstacle.color.r = 0.5;
	}
	new_obstacle.color.a = 0.5;
	new_obstacle.id = id;
	new_obstacle.lifetime = ros::Duration(1.0);

	// geometry_msgs::PoseStamped pose_in;
	// pose_in.header = new_obstacle.header;
	// pose_in.pose = new_obstacle.pose;
	// geometry_msgs::PoseStamped pose_out;
	// listener.transformPose("ZOE3/os_sensor", pose_in, pose_out);

	// new_obstacle.pose = pose_out.pose;
	// new_obstacle.header = pose_out.header;
	obstacles.markers.clear();
	obstacles.markers.push_back(new_obstacle);



	}

};



/* Callback called when the client receives a CONNACK message from the broker. */
void on_connect(struct mosquitto *mosq, void *obj, int reason_code)
{
	int rcx;
	int rcy;
	int rcf;
	/* Print out the connection result. mosquitto_connack_string() produces an
	 * appropriate string for MQTT v3.x clients, the equivalent for MQTT v5.0
	 * clients is mosquitto_reason_string().
	 */
	printf("on_connect: %s\n", mosquitto_connack_string(reason_code));
	if(reason_code != 0){
		/* If the connection fails for any reason, we don't want to keep on
		 * retrying in this example, so disconnect. Without this, the client
		 * will attempt to reconnect. */
		mosquitto_disconnect(mosq);
	}

	/* Making subscriptions in the on_connect() callback means that if the
	 * connection drops and is automatically resumed by the client, then the
	 * subscriptions will be recreated when the client reconnects. */
	rcx = mosquitto_subscribe(mosq, NULL, "RSU/#", 1);
	// rcy = mosquitto_subscribe(mosq, NULL, "y_global", 1);
	// rcf = mosquitto_subscribe(mosq, NULL, "frame_id", 1);
	if(rcx != MOSQ_ERR_SUCCESS){
		fprintf(stderr, "Error subscribing: %s\n", mosquitto_strerror(rcx));
		/* We might as well disconnect if we were unable to subscribe */
		mosquitto_disconnect(mosq);
	// if(rcy != MOSQ_ERR_SUCCESS){
	// 	fprintf(stderr, "Error subscribing: %s\n", mosquitto_strerror(rcy));
	// 	/* We might as well disconnect if we were unable to subscribe */
	// 	mosquitto_disconnect(mosq);
	// 	}
	}
}


/* Callback called when the broker sends a SUBACK in response to a SUBSCRIBE. */
void on_subscribe(struct mosquitto *mosq, void *obj, int mid, int qos_count, const int *granted_qos)
{
	int i;
	bool have_subscription = false;
	

	/* In this example we only subscribe to a single topic at once, but a
	 * SUBSCRIBE can contain many topics at once, so this is one way to check
	 * them all. */
	for(i=0; i<qos_count; i++){
		printf("on_subscribe: %d:granted qos = %d\n", i, granted_qos[i]);
		if(granted_qos[i] <= 2){
			have_subscription = true;
		}
	}
	if(have_subscription == false){
		/* The broker rejected all of our subscriptions, we know we only sent
		 * the one SUBSCRIBE, so there is no point remaining connected. */
		fprintf(stderr, "Error: All subscriptions rejected.\n");
		mosquitto_disconnect(mosq);
		
	}
}


/* Callback called when the client receives a message. */
void on_message(struct mosquitto *mosq, void *obj, const struct mosquitto_message *msg)
{
	/* This blindly prints the payload, but the payload can be anything so take care. */
	string topic =  msg->topic;
	string msg_to_read = (char *)msg->payload;
	MarkerArrayModifier modifier(topic, msg_to_read, obstacles);
}


int main(int argc, char *argv[])
{
	ros::init(argc, argv, "icars_mosquitto_subscriber");
	ros::NodeHandle n;
	std::string broker;
	n.param("broker", broker , std::string("130.66.64.112"));

	ros::Publisher obstacleArray_pub = n.advertise<visualization_msgs::MarkerArray>("markersArray", 5);
    
    	
    	
    	
	struct mosquitto *mosq;
	int rcx;

	/* Required before calling other mosquitto functions */
	mosquitto_lib_init();

	/* Create a new client instance.
	 * id = NULL -> ask the broker to generate a client id for us
	 * clean session = true -> the broker should remove old sessions when we connect
	 * obj = NULL -> we aren't passing any of our private data for callbacks
	 */
	mosq = mosquitto_new(NULL, true, NULL);
	if(mosq == NULL){
		fprintf(stderr, "Error: Out of memory.\n");
		return 1;
	}

	/* Configure callbacks. This should be done before connecting ideally. */
	mosquitto_connect_callback_set(mosq, on_connect);
	mosquitto_subscribe_callback_set(mosq, on_subscribe);
	mosquitto_message_callback_set(mosq, on_message);

	/* Connect to test.mosquitto.org on port 1883, with a keepalive of 60 seconds.
	 * This call makes the socket connection only, it does not complete the MQTT
	 * CONNECT/CONNACK flow, you should use mosquitto_loop_start() or
	 * mosquitto_loop_forever() for processing net traffic. */
	rcx = mosquitto_connect(mosq, broker.c_str(), 1883, 60);
	if(rcx != MOSQ_ERR_SUCCESS){
		mosquitto_destroy(mosq);
		fprintf(stderr, "Error: %s\n", mosquitto_strerror(rcx));
		return 1;
	}
	

	/* Run the network loop in a blocking call. The only thing we do in this
	 * example is to print incoming messages, so a blocking call here is fine.
	 *
	 * This call will continue forever, carrying automatic reconnections if
	 * necessary, until the user calls mosquitto_disconnect().
	 */
	ros::Rate loop_rate(10);

	while (ros::ok())
	{
		mosquitto_loop(mosq,-1,1);
		// visualization_msgs::Marker list_obstacles;
		obstacleArray_pub.publish(obstacles);
		// obstacle_pub.publish(obstacle);
		ros::spinOnce();
	}
	mosquitto_lib_cleanup();
	return 0;
}





