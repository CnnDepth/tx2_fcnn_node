#include <tx2_fcnn_node/ros_fcnn_inference.hpp>
#include <tx2_fcnn_node/default_params.hpp>

#include <ros/package.h>

#include <cudaMappedMemory.h>
#include <cudaUtility.h>
#include <cudaRGB.h>

#include <slicePlugin.h>
#include <interleavingPlugin.h>
#include <upsamplingPlugin.h>

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

PluginFieldCollection NearestNeighborUpsamplingPluginCreator::mFC{};
std::vector<PluginField> NearestNeighborUpsamplingPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN( NearestNeighborUpsamplingPluginCreator );

PluginFieldCollection StridedSlicePluginCreator::mFC{};
std::vector<PluginField> StridedSlicePluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN( StridedSlicePluginCreator );

PluginFieldCollection InterleavingPluginCreator::mFC{};
std::vector<PluginField> InterleavingPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN( InterleavingPluginCreator );

RosFcnnInference::RosFcnnInference( ros::NodeHandle& _nh )
: mNodeHandle( _nh )
, mImageTransport( _nh )
{
    ROS_INFO( "Starting tx2_fcnn_node" );
    this->initializeParameters();
    this->initializePublishers();
    
    if( this->mUseInternalCamera )
    {
        this->initializeInputSource();
    }
    this->initializeEngine();
}

RosFcnnInference::~RosFcnnInference()
{
    deallocateCudaMemory();
    destroyEngine();
    if( this->mUseInternalCamera )
    {
        destroyCamera();
    }
}

void RosFcnnInference::run()
{
    //TODO: Set as parameter 
    ros::Rate loopRate(120);

    while( this->mNodeHandle.ok() )
    {
        ros::spinOnce();
    }
}

void RosFcnnInference::initializeParameters()
{
    ROS_INFO( "Initializing parameters" );
    
    this->mNodeHandle.param<int>( "input_width", this->mInputImageWidth, DEFAULT_INPUT_WIDTH );
    if( this->mInputImageWidth < 0 )
    {
        ROS_WARN( "input_width is less than 0. Using default value: %d", DEFAULT_INPUT_WIDTH );
        this->mInputImageWidth = DEFAULT_INPUT_WIDTH;
    }
    
    this->mNodeHandle.param<int>( "input_height", this->mInputImageHeight, DEFAULT_INPUT_HEIGHT );
    if( this->mInputImageHeight < 0 )
    {
        ROS_WARN( "input_width is less than 0. Using default value: %d", DEFAULT_INPUT_HEIGHT );
        this->mInputImageHeight = DEFAULT_INPUT_HEIGHT;
    }

    this->mNodeHandle.param<int>( "camera_mode", this->mCameraMode, DEFAULT_CAMERA_MODE );
    if( this->mCameraMode < -1 )
    {
        ROS_WARN( "camera_mode is less then -1. Using default value: %d", DEFAULT_CAMERA_MODE );
    }


    this->mNodeHandle.param<bool>( "use_camera", this->mUseInternalCamera, DEFAULT_USE_CAMERA );

    this->mNodeHandle.param<std::string>( "camera_link", this->mCameraLink, DEFAULT_CAMERA_LINK );
    this->mNodeHandle.param<std::string>( "depth_link", this->mDepthLink, DEFAULT_DEPTH_LINK );

    this->mNodeHandle.param<std::string>( "engine_name", this->mEngineName, DEFAULT_ENGINE_NAME );
    this->mNodeHandle.param<std::string>( "calib_name", this->mCalibName, DEFAULT_CALIB_NAME );

    this->mNodeHandle.param<std::string>( "input_name", this->mEngineInputName, DEFAULT_INPUT_NAME );
    this->mNodeHandle.param<std::string>( "output_name", this->mEngineOutputName, DEFAULT_OUTPUT_NAME );

}

/* void RosFcnnInference::initializeSubscribers()
{

}*/

void RosFcnnInference::initializePublishers()
{
    ROS_INFO( "Initializing publishers" );
    this->mImagePublisher     = this->mImageTransport.advertise( "/rgb/image", 1 );
    this->mDepthPublisher     = this->mImageTransport.advertise( "/depth/image", 1 );

    this->mImageInfoPublisher = this->mNodeHandle.advertise<sensor_msgs::CameraInfo>( "/rgb/camera_info", 1 );
    this->mDepthInfoPublisher = this->mNodeHandle.advertise<sensor_msgs::CameraInfo>( "/depth/camera_info", 1 );
}

void RosFcnnInference::initializeInputSource()
{
    ROS_INFO( "Initializing camera" );
    this->mCamera = gstCamera::Create( this->mInputImageWidth, this->mInputImageHeight, this->mCameraMode );

    if( !this->mCamera )
    {
        ROS_FATAL( "Failed to initialize video device" );

        return;
    }

    if( !this->mCamera->Open() )
    {
        ROS_FATAL( "Failed to open camera" );

        return;
    }
}

//WARNIGN: Probably unsafe. Not sure if buffer is needed
void RosFcnnInference::initializeEngine()
{
    ROS_INFO( "Creating engine" );

    Logger gLogger;

    std::ifstream engineModel( std::string( ros::package::getPath( PACKAGE_NAME ) ) + "/engine/" + this->mEngineName );
    if( !engineModel )
    {
        ROS_FATAL( "Failed to open engine file" );
    }

    std::vector<unsigned char> buffer( std::istreambuf_iterator<char>( engineModel ), {} );
    std::size_t                modelSize = buffer.size() * sizeof( unsigned char );
    
    //const int outputIndex   = this->mCudaEngine->getBindingIndex( this->mEngineInputName.c_str() );
    this->mRuntimeInfer     = nvinfer1::createInferRuntime( gLogger );
    this->mCudaEngine       = this->mRuntimeInfer->deserializeCudaEngine( buffer.data(), modelSize, nullptr );
    this->mExecutionContext = this->mCudaEngine->createExecutionContext();

    if( !this->mExecutionContext )
    {
        ROS_FATAL( "Failed to create execution context" );
    }

    // Get input size
    const int       inputIndex   = this->mCudaEngine->getBindingIndex( this->mEngineInputName.c_str() );
    nvinfer1::Dims  inputDims    = this->mCudaEngine->getBindingDimensions( inputIndex );

    this->mInputSize        = DIMS_C( inputDims ) * DIMS_H( inputDims ) * DIMS_W( inputDims ) * sizeof(float);
    this->mInputSizeUint8   = DIMS_C( inputDims ) * DIMS_H( inputDims ) * DIMS_W( inputDims ) * sizeof(uint8_t);
    
    // Get output size
    const int       outputIndex  = this->mCudaEngine->getBindingIndex( this->mEngineOutputName.c_str() );
    nvinfer1::Dims  outputDims   = this->mCudaEngine->getBindingDimensions( outputIndex );
    
    this->mOutputSize      = DIMS_C( outputDims ) * DIMS_H( outputDims ) * DIMS_W( outputDims ) * sizeof(float);

    ROS_INFO( "DONE!" );
}

void RosFcnnInference::allocateCudaMemory()
{

}

void RosFcnnInference::deallocateCudaMemory()
{

}

void RosFcnnInference::destroyEngine()
{

}

void RosFcnnInference::destroyCamera()
{

}

void RosFcnnInference::grabImageAndPreprocess()
{

}
