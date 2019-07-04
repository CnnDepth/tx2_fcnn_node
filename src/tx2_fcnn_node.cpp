#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float64.h>
#include <camera_calibration_parsers/parse.h>

#include <tx2_fcnn_node/ros_fcnn_inference.hpp>

#include <chrono>
#include <fstream>

#include <NvInfer.h>
#include <gstCamera.h>
#include <cudaMappedMemory.h>
#include <cudaUtility.h>
#include <cudaRGB.h>

#include "slicePlugin.h"
#include "interleavingPlugin.h"
#include "upsamplingPlugin.h"

#include <iostream>
#include <opencv2/highgui.hpp>

#include <tx2_fcnn_node/ros_utils.hpp>

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

const std::string PACKAGE_NAME = "tx2_fcnn_node";

const int DEFAULT_CAMERA_HEIGHT = 240;
const int DEFAULT_CAMERA_WIDTH  = 320;

Logger gLogger;

cudaError_t cudaPreImageNetMean(float4* input, size_t inputWidth, size_t inputHeight,
				             float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream);



int main( int argc, char** argv )
{
  ros::init(argc, argv, PACKAGE_NAME);
  ros::NodeHandle nh("~");

  RosFcnnInference nnInf( nh );
  nnInf.run();

  return 0;
}
