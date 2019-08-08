#include <ros/ros.h>
#include <tx2_fcnn_node/ros_fcnn_inference.hpp>

const std::string PACKAGE_NAME = "tx2_fcnn_node";

int main( int argc, char** argv )
{
  ros::init(argc, argv, PACKAGE_NAME);
  ros::NodeHandle nh("~");

  RosFcnnInference nnInf( nh );
  nnInf.run();

  return 0;
}

