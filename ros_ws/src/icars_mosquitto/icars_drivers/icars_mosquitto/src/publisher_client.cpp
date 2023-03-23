#include <stdio.h>
#include <string.h>
#include <mosquitto.h>
#include <cstring>
#include <iostream>
#include <string>
#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"

using namespace std;

void chatterCallback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
  float Arr[90];
  int i = 0;
	// print all the remaining numbers
	for(std::vector<float>::const_iterator it = msg->data.begin(); it != msg->data.end(); ++it)
	{
		Arr[i] = *it;
		i++;
	}
  cout<<Arr[0]<<"\n"<<endl;

}

int main(int argc, char *argv[]) 
{
	ros::init(argc, argv, "icars_mosquitto_publisher");
    ros::NodeHandle n;
    
    std::string broker;
    n.param("broker", broker , std::string("130.66.64.112"));
    int port = 1883;
    string agent = "RSU";
    int nb_bb = 5;
    string id_bb[nb_bb];
    for(int n=1; n<nb_bb+1; n++){
    	id_bb[n-1] = to_string(n);
    }
    string subtopics[] = {"cx", "cy", "cz","dx","dy","dz", "heading","class"}; 
    string msgs1[] = {"1.0", "2.0", "0.5", "0.1","0.1","0.1", "0.5", "car"};
    string msgs2[] = {"3.0", "4.0", "0.5", "0.1","0.1","0.1", "0.5", "car"};
    string msgs3[] = {"1.0", "2.0", "0.5", "0.1","0.1","0.1", "0.5", "car"};
    string msgs4[] = {"1.0", "2.0", "0.5", "0.1","0.1","0.1", "0.5", "car"};
    string msgs5[] = {"1.0", "2.0", "0.5", "0.1","0.1","0.1", "0.5", "car"};
    
    
    struct mosquitto *client;
    int ret = 0;
    
    // initialise la bibliothèque
    mosquitto_lib_init();
    
    // crée un client    
    // parametres : generate an id, create a clean session, no callback param 
    client = mosquitto_new(NULL, true, NULL);
    
    ret = mosquitto_connect(client, broker.c_str(), port, 60); // connexion persistante
    if (ret != MOSQ_ERR_SUCCESS) 
    {
        perror("mosquitto_connect");
        return(ret);        
    }
    
    ros::Rate loop_rate(1);
    ros::Subscriber chatter_sub = n.subscribe("obstacle_list", 1000, chatterCallback);
    while (ros::ok())
  	{ 

    
    // publie un message dans le sujet
    string id_flag = "1";
    string msg;
    for(string id: id_bb){
     printf("-------------------------------");
    	// printf("%s \n", id.c_str());
    	string topic_name = agent + '/' + id + '/' ;
	    for(int idx = 0 ; idx<sizeof(msgs1)/sizeof(msgs1[0]); idx++){
	    	
	    	msg = msg + '/' + msgs1[idx];
	    	printf("%s %s \n", topic_name.c_str(), msg.c_str());

		  }
    mosquitto_publish(client, NULL, topic_name.c_str(), msg.length(),msg.c_str(), 0, false); 	
    msg = "";
	}
	ros::spinOnce();
	loop_rate.sleep();
	
	}
	
    // détruit le client
    mosquitto_destroy(client);
    
    // libère les ressources MQTT
    mosquitto_lib_cleanup();
    
    return 0;
}
