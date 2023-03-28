#include <stdio.h>
#include <string.h>
#include <mosquitto.h>
#include <cstring>
#include <iostream>
#include <string>
#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"

using namespace std;

class Bbox_Subscriber
{
public:
  Bbox_Subscriber(ros::NodeHandle &nh)
  {

    sub_ = nh.subscribe("obstacle_list", 20, &Bbox_Subscriber::callback, this);
  }
  vector<vector<string> > msgs;
private:
  ros::Subscriber sub_;
  

  void callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
  {
  msgs.clear();
	for(int i=0; i<(msg->data).size();i=i+9)
	{
    vector<string> bbox_feat;
    bbox_feat.reserve(9);
    int j=0;
    while(j<9){
      if(j==7){bbox_feat.push_back(to_string(int(msg->data[j+i])));}
      else{
      bbox_feat.push_back(to_string(msg->data[j+i]));
      }
      j++;
    }
    msgs.push_back(bbox_feat);
	}
  }
};


int main(int argc, char *argv[]) 
{
	ros::init(argc, argv, "icars_mosquitto_publisher");
    ros::NodeHandle n;
    
    std::string broker;
    n.param("broker", broker , std::string("130.66.64.112"));
    int port = 1883;
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
    
    ros::Rate loop_rate(10);
    // crée un subscriber du node de det&tracking 
    Bbox_Subscriber subscriber(n);
    while (ros::ok())
  	{ 
    // publie un message dans le sujet
    // printf("-------------------- \n");
    if (subscriber.msgs.size()>0){
      for(int j=0; j<subscriber.msgs.size(); j++){
        string msg_to_publish;
        string topic = "RSU/" + subscriber.msgs[j][7];
        for(int i=0; i<subscriber.msgs[j].size(); i++){
          if(i!=7){
          msg_to_publish = msg_to_publish + subscriber.msgs[j][i] + "/";
          }
        }
        // cout<<"topic is:"<< topic <<endl;
        // cout<<"msg is:"<< msg_to_publish <<endl;
        mosquitto_publish(client, NULL, topic.c_str(), msg_to_publish.length(),msg_to_publish.c_str(), 0, false);
    } 
    }
    // printf("--------------------");
 
	ros::spinOnce();
	loop_rate.sleep();
	
	}
	
    // détruit le client
    mosquitto_destroy(client);
    
    // libère les ressources MQTT
    mosquitto_lib_cleanup();
    
    return 0;
}
