#include <stdio.h>
#include <string.h>
#include <mosquitto.h>
#include <iostream>

#include "ros/ros.h"


#include "visualization_msgs/MarkerArray.h"
#include "visualization_msgs/Marker.h"
#include <tf2/LinearMath/Quaternion.h>

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

/*icars_msgs::PathObstacle obstacle;
icars_msgs::PathObstacleList obstaclelist;*/
visualization_msgs::MarkerArray obstacles;
visualization_msgs::Marker obstacle;

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
	rcy = mosquitto_subscribe(mosq, NULL, "y_global", 1);
	rcf = mosquitto_subscribe(mosq, NULL, "frame_id", 1);
	if(rcx != MOSQ_ERR_SUCCESS){
		fprintf(stderr, "Error subscribing: %s\n", mosquitto_strerror(rcx));
		/* We might as well disconnect if we were unable to subscribe */
		mosquitto_disconnect(mosq);
	if(rcy != MOSQ_ERR_SUCCESS){
		fprintf(stderr, "Error subscribing: %s\n", mosquitto_strerror(rcy));
		/* We might as well disconnect if we were unable to subscribe */
		mosquitto_disconnect(mosq);
		}
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
	printf("%s %d %s\n", msg->topic, msg->qos, (char *)msg->payload);
	
	/* char* topic1 = "RSU/0/x_global";
	if(strcmp(msg->topic, topic1) == 0){
		obstacle.x_global = atof((char *)msg->payload);
	}
	char* topic2 = "y_global";
	if(strcmp(msg->topic, topic2) == 0){
		obstacle.y_global = atof((char *)msg->payload);
	}
	char* topic3 = "frame_id";
	if(strcmp(msg->topic, topic3) == 0){
		obstacle.header.frame_id = (char *)msg->payload;
	}
	
	obstaclelist.obst_list.push_back(obstacle);*/
	// RVIZ visualization
	
	// ------------------------------------------Topics segmentation
    	char *topic_ptr = strtok(msg->topic, "/");
    	int i=0;
    	char *agent;
    	char *id ;
    	char *topic_name ;
    	while (topic_ptr != NULL)  
    	{
    	switch (i){
    		case 0:
    			agent = topic_ptr;
    		case 1:
    			id = topic_ptr;
        topic_ptr = strtok (NULL, "/"); 
        i++;
    	}
    	}
    	
    	// ------------------------------------------Msg segmentation
    	char *msg_ptr = strtok((char *)msg->payload, "/");
    	int j=0;
    	float cx, cy, cz, dx, dy, dz, heading;
    	char *classe; 
    	char * endPtr;
    	while (msg_ptr!= NULL)  
    	{
    	switch (j){
    		case 0:
    			cx = atof(msg_ptr);
    			cout<<cx<<endl;
    		case 1:
    			cy = atof(msg_ptr);
    		case 2:
    			cz = atof(msg_ptr);
    		case 3:
    			dx = atof(msg_ptr);
    		case 4:
    			dy = atof(msg_ptr);
    		case 5:
    			dz = atof(msg_ptr);
    		case 6:
    			heading = atof(msg_ptr);
    		case 7:
    			classe = msg_ptr;
        msg_ptr = strtok (NULL, "/"); 
        j++;
    	}
    	}
    	//printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %s \n", cx, cy, cz, dx, dy, dz, heading, classe);
    	visualization_msgs::Marker new_obstacle;
    	new_obstacle.pose.position.x = cx;
    	new_obstacle.pose.position.y = cy;
    	new_obstacle.pose.position.z = cz;
    	new_obstacle.scale.x = dx;
    	new_obstacle.scale.y = dy;
    	new_obstacle.scale.z = dz;
    	tf2::Quaternion q_heading;
    	q_heading.setRPY( 0, 0, heading);
    	new_obstacle.pose.orientation.x = q_heading.getX();
    	new_obstacle.pose.orientation.y = q_heading.getY();
    	new_obstacle.pose.orientation.z = q_heading.getZ();
    	new_obstacle.pose.orientation.w = q_heading.getW();
    	//new_obstacle.pose.position.x = classe;
    	
    	/*for (visualization_msgs::Marker obstacle : obstacles.markers) 
    	{
    		if ( obstacle.id == atoi(id) ) 
    		{
			
    			return;
    		}  
    	}*/
    	/*for(int len_obstacle = 0; len_obstacle<obstacles.markers.size();len_obstacle++ ){
    		printf("len of obstacles %d \n", obstacles.markers[len_obstacle].id);
    		if(atoi(id) == obstacles.markers[len_obstacle].id){return;}
 	}

 	obstacle.ns = "my_namespace";
	obstacle.id = atoi(id);
    	char *topicn = "cx";
    	if (strcmp(topic_name, topicn) == 0) {obstacle.pose.position.x = atof((char *)msg->payload);}
    	topicn = "cy";
    	if (strcmp(topic_name, topicn) == 0) {obstacle.pose.position.y = atof((char *)msg->payload);}
    	topicn = "cz";
    	if(strcmp(topic_name, topicn) == 0) {obstacle.pose.position.z = atof((char *)msg->payload); }
    	topicn = "dx";
    	if(strcmp(topic_name, topicn) == 0) {obstacle.scale.x = atof((char *)msg->payload); }
    	topicn = "dy";
    	if(strcmp(topic_name, topicn) == 0) {obstacle.scale.y = atof((char *)msg->payload); }
    	topicn = "dz";
    	if(strcmp(topic_name, topicn) == 0) {obstacle.scale.z = atof((char *)msg->payload);}
    	topicn = "heading";
    	tf2::Quaternion q_heading;
    	if(strcmp(topic_name, topicn) == 0) {q_heading.setRPY( 0, 0, atof((char *)msg->payload));
    	obstacle.pose.orientation.x = q_heading.getX();
    	obstacle.pose.orientation.y = q_heading.getY();
    	obstacle.pose.orientation.z = q_heading.getZ();
    	obstacle.pose.orientation.w = q_heading.getW();}
    	topicn = "class";
    	//if(strcmp(topic_name, topicn) == 0) {obstacle.color.a = atof((char *)msg->payload); printf("enter the void");}
    	*/
    	new_obstacle.header.frame_id = "base_link";
	new_obstacle.type = 1;
	new_obstacle.color.r = 1.0;
	new_obstacle.color.a = 1.0;
	
    	obstacle = new_obstacle;
	//obstacle.action = visualization_msgs::Marker::MODIFY;
	//visualization_msgs::Marker new_obstacle;
	//obstacles.markers.push_back(new_obstacle);
	
}


int main(int argc, char *argv[])
{
	ros::init(argc, argv, "icars_mosquitto_subscriber");
	ros::NodeHandle n;
    	std::string broker;
    	n.param("broker", broker , std::string("130.66.64.112"));
    	
    	ros::Publisher chatter_pub = n.advertise<visualization_msgs::Marker>("pathobstacle", 1000);
    	
    	
    	
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
	
	while (ros::ok())
	{
		mosquitto_loop(mosq,-1,1);
		chatter_pub.publish(obstacle);
		ros::spinOnce();
	}
	mosquitto_lib_cleanup();
	return 0;
}





