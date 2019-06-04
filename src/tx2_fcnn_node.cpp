#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float64.h>

#include <chrono>
#include <fstream>

#include <NvInfer.h>
#include <gstCamera.h>
#include <cudaMappedMemory.h>
#include <cudaUtility.h>
#include <cudaRGB.h>

#include "../Thirdparty/fcrn-camera/upsampling/plugin.h"

#include <iostream>
#include <opencv2/highgui.hpp>

#include <tx2_fcnn_node/ros_utils.hpp>

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

const int DEFAULT_CAMERA_HEIGHT = 640;

/*class Logger : public nvinfer1::ILogger
{
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override
  {
    std::cout << msg << std::endl;
  }
};*/

Logger gLogger;

//nvinfer1::ILogger gLogger;


cudaError_t cudaPreImageNetMean(float4* input, size_t inputWidth, size_t inputHeight,
				             float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream);


PluginFieldCollection NearestNeighborUpsamplingPluginCreator::mFC{};
std::vector<PluginField> NearestNeighborUpsamplingPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN( NearestNeighborUpsamplingPluginCreator );

int main( int argc, char** argv )
{
  ros::init(argc, argv, "tx2_fcnn_node");
  ros::NodeHandle nh;


  int inputCameraHeight;
  int inputCameraWidth;
  ros::param::param<int>( "camera_height", inputCameraHeight, DEFAULT_CAMERA_HEIGHT );

  ROS_INFO("Starting tx2_fcnn_node...");
//===========================
  gstCamera* camera = gstCamera::Create( 640, 480, -1 );
  
  if( !camera )
  {
    ROS_ERROR( "Failed to initialize video device" );

    return -1;
  }

  ROS_INFO( "Camera initialized:" );
  ROS_INFO( "     width: %d", camera->GetWidth() );
  ROS_INFO( "    height: %d", camera->GetHeight() );
  ROS_INFO( "     depth: %d", camera->GetPixelDepth() );

//  std::ifstream engineModel( "/media/sd/kmouraviev/engines/nonbt_engine_shortcuts_320x240.trt", std::ios::binary );
  std::ifstream engineModel( "/media/sd/kmouraviev/engines/trt_engine_fullsize_5x5.trt" );
  std::vector<unsigned char> buffer( std::istreambuf_iterator<char>( engineModel ), {} );
  std::size_t modelSize = buffer.size() * sizeof( unsigned char );

  nvinfer1::IRuntime* runtime          = nvinfer1::createInferRuntime( gLogger );
  nvinfer1::ICudaEngine* engine        = runtime->deserializeCudaEngine( buffer.data(), modelSize, nullptr );
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();

  if( !context )
  {
    ROS_ERROR( "Failed to create execution context" );

    return -1;
  }

  const int inputIndex     = engine->getBindingIndex( "tf/Placeholder" );
  nvinfer1::Dims inputDims = engine->getBindingDimensions( inputIndex );

  std::size_t inputSize      = DIMS_C( inputDims ) * DIMS_H( inputDims ) * DIMS_W( inputDims ) * sizeof(float);
  std::size_t inputSizeUint8 = DIMS_C( inputDims ) * DIMS_H( inputDims ) * DIMS_W( inputDims ) * sizeof(uint8_t);

  float* outImageCPU  = nullptr;
  float* outImageCUDA = nullptr;

  const int outputIndex     = engine->getBindingIndex( "tf/Reshape" );
  nvinfer1::Dims outputDims = engine->getBindingDimensions( outputIndex );
  std::size_t outputSize    = DIMS_C( outputDims ) * DIMS_H( outputDims ) * DIMS_W( outputDims ) * sizeof(float);

  void* imgRGBCPU = NULL;
  void* imgRGB = NULL;
  if( !cudaAllocMapped((void**)&imgRGBCPU, (void**)&imgRGB, inputSize) )
  {
    ROS_ERROR("Failed to alloc CUDA mapped memory for RGB image");
    return false;
  }

  
  if( !cudaAllocMapped( (void**)&outImageCPU, (void**)&outImageCUDA, outputSize ) )
  {
    ROS_ERROR( "Failed to allocate CUDA memory for output" );

    return -1;
  }

  cudaStream_t stream = nullptr;
  CUDA_FAILED( cudaStreamCreateWithFlags( &stream, cudaStreamDefault ) );


  void* rgb8ImageCPU  = NULL;
  void* rgb8ImageCUDA = NULL;
  if( !cudaAllocMapped( (void**)&rgb8ImageCPU, (void**)&rgb8ImageCUDA, inputSizeUint8 ) )
  {
    ROS_ERROR( "Failed to allocate CUDA mapped memory for RGB image" );

    return -1;
  }

  void* dividedDepth = NULL;
  void* dividedDepthCPU = NULL;
  if( !cudaAllocMapped((void**)&dividedDepthCPU, (void**)&dividedDepth, outputSize) )
  {
    ROS_ERROR("failed to alloc CUDA mapped memory for RGB image, %zu bytes\n");
    return false;
  }


  if( !camera->Open() )
  {
    ROS_ERROR( "Failed to open camera" );

    return -1;
  }

  

//==========================



//  std::string pipeline = get_tegra_pipeline( 1280, 720, 30 );
//  cv::VideoCapture vCap(0);
//  cv::VideoCapture vCap( pipeline, cv::CAP_GSTREAMER );

//   cv::VideoCapture vCap("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate=120/1, format=(string)NV12 ! nvvidconv ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! appsink", cv::CAP_GSTREAMER);

//  cv::VideoCapture vCap("v4l2src device=/dev/video1 ! video/x-raw, width=(int)3840, height=(int)2160,format=(string)I420, framerate=(fraction)30/1 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
//  ROS_INFO( "Using pipeline: %s", pipeline.c_str() );

/*
  if( !vCap.isOpened() )
  {
    ROS_INFO( "Error!" );
    vCap.release();
    return -1;
  }
*/
  ros::Publisher ciPub = nh.advertise<sensor_msgs::CameraInfo>( "/rgb/camera_info", 1 );
  sensor_msgs::CameraInfo ciMsg;
  ciMsg.header = std_msgs::Header();
  ciMsg.header.frame_id = "camera_link";
  ciMsg.height = 480;
  ciMsg.width  = 640;
  ciMsg.distortion_model = "plumb_bob";
//  double distor[5] = {0.262383, -0.953104, -0.005358, 0.002628, 1.163314};
//  ciMsg.D = {0.262383, -0.953104, -0.005358, 0.002628, 1.163314};
//  double k[9] = {517.306408, 0, 318.643040, 0, 517.469215, 255.313989, 0, 0, 1};
//  ciMsg.K = {517.306408, 0, 318.643040, 0, 517.469215, 255.313989, 0, 0, 1};
//  ciMsg.R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
//  ciMsg.P = { 535.4, 0, 320.1, 0, 0, 539.2, 247.05, 0.0, 0.0, 1.0 };

  ciMsg.D = {0.11754636857648042, -0.20796214251890827, 0.003951824147288686, 0.0011981225998639721, 0.0};
//  double k[9] = {517.306408, 0, 318.643040, 0, 517.469215, 255.313989, 0, 0, 1};
  ciMsg.K = {239.12800192487694, 0.0, 164.5580937136966, 0.0, 319.0746579524651, 121.00594783717585, 0.0, 0.0, 1.0};
  ciMsg.R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  ciMsg.P = { 241.8467254638672, 0.0, 165.24418493136181, 0.0, 0.0, 324.6019287109375, 121.7070042868836, 0.0, 0.0, 0.0, 1.0, 0.0 };


  ros::Publisher diPub = nh.advertise<sensor_msgs::CameraInfo>( "/depth/camera_info", 1 );
  sensor_msgs::CameraInfo diMsg;
  diMsg.header = std_msgs::Header();
  diMsg.header.frame_id = "base_scan";
  diMsg.height = 480;
  diMsg.width  = 640;
  diMsg.distortion_model = "plumb_bob";
//  double distor[5] = {0.262383, -0.953104, -0.005358, 0.002628, 1.163314};
  diMsg.D = {0.11754636857648042, -0.20796214251890827, 0.003951824147288686, 0.0011981225998639721, 0.0};
//  double k[9] = {517.306408, 0, 318.643040, 0, 517.469215, 255.313989, 0, 0, 1};
  diMsg.K = {239.12800192487694, 0.0, 164.5580937136966, 0.0, 319.0746579524651, 121.00594783717585, 0.0, 0.0, 1.0};
  diMsg.R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  diMsg.P = { 241.8467254638672, 0.0, 165.24418493136181, 0.0, 0.0, 324.6019287109375, 121.7070042868836, 0.0, 0.0, 0.0, 1.0, 0.0 };
  


  image_transport::ImageTransport it(nh);
  image_transport::Publisher      pub  = it.advertise( "/rgb/image", 1 );
  image_transport::Publisher  depthPub = it.advertise( "/depth/image", 1 );

  cv::Mat frame;
  
//  ImageConverter imgConv(nh);
  std::chrono::time_point<std::chrono::system_clock> start, end;


  ros::Rate loop_rate(120);
  start = std::chrono::system_clock::now();

  while( nh.ok() )
  {
/*    start = std::chrono::system_clock::now();

    vCap >> frame;

//    imgConv.publish( frame );
//    cv::imshow("asd", frame);
//    cv::waitKey(1);

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage( std_msgs::Header(), "bgr8", frame ).toImageMsg();


    pub.publish(msg);

    ros::spinOnce();
    loop_rate.sleep();

    end = std::chrono::system_clock::now();
    int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    ROS_INFO( "vCap Time: %d", elapsed_seconds );
*/
    void* imgCPU  = NULL;
    void* imgCUDA = NULL;
    void* imgRGBA = NULL;

    if( !camera->Capture( &imgCPU, &imgCUDA, 1000 ) )
    {
      ROS_WARN( "Failed to capture frame" );
    }

    if( !camera->ConvertRGBA( imgCUDA, &imgRGBA, true ) )
    {
      ROS_WARN( "Failed to convert from NV12 to RGBA" );
    }

   CUDA( cudaRGBA32ToRGB8( (float4*)imgRGBA, (uchar3*)rgb8ImageCUDA, camera->GetWidth(), camera->GetHeight() ) );

   cv::Mat rosImage( camera->GetHeight(), camera->GetWidth() , CV_8UC3, rgb8ImageCUDA );
   sensor_msgs::ImagePtr msg = cv_bridge::CvImage( std_msgs::Header(), "rgb8", rosImage ).toImageMsg();







    const float3 mean_value = make_float3( 123.0, 115.0, 101.0 );

    if( CUDA_FAILED( cudaPreImageNetMean( (float4*)imgRGBA, 640, 480, (float*)imgRGB, 640, 480, mean_value, stream ) ) )
    {
      ROS_ERROR( "Failed to preprocess" );

      return -1;
    }

    void* bindings[] = {imgRGB, outImageCUDA};
    context->execute( 1, bindings );

//    CUDA_FAILED( cudaDivideByMaxValue( outImageCUDA, (float*)dividedDepth, camera->GetWidth(), camera->GetHeight(), 10 ) );
    cv::Mat outDepth( camera->GetHeight(), camera->GetWidth(), CV_32FC1, outImageCUDA );

//    std::cout << outDepth << std::endl;

    double min, max;
    cv::minMaxLoc( outDepth, &min, &max );
    ROS_INFO( "-- min: %d max: %d", min, max  );

//    outDepth.convertTo( outDepth, CV_8UC1, 255.0/(max-min), -min * ( 255.0/(max-min) ));

    sensor_msgs::ImagePtr depthMsg = cv_bridge::CvImage( std_msgs::Header(), sensor_msgs::image_encodings::TYPE_32FC1, outDepth ).toImageMsg();

    ros::Time timestamp = ros::Time::now();

    msg->header.frame_id = "camera_link";
    depthMsg->header.frame_id = "base_scan";

    msg->header.stamp = timestamp;
    pub.publish(msg);

    depthMsg->header.stamp = timestamp;
    depthPub.publish( depthMsg );

    ciMsg.header.stamp = timestamp;
    ciPub.publish( ciMsg );

    diMsg.header.stamp = timestamp;
    diPub.publish( diMsg );

    ros::spinOnce();
    loop_rate.sleep();
  }

//  vCap.release();

  return 0;
}
