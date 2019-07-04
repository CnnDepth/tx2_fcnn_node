#include <tx2_fcnn_node/ros_fcnn_inference.hpp>
#include <tx2_fcnn_node/default_params.hpp>

#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <camera_calibration_parsers/parse.h>

#include <opencv2/opencv.hpp>

#include <cudaMappedMemory.h>
#include <cudaRGB.h>
#include <preprocessRGB.h>

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
    this->setOutputCameraInfo();
    if( this->mUseInternalCamera )
    {
        this->initializeInputSource();
    }
    this->initializeEngine();
    this->allocateCudaMemory();
}

RosFcnnInference::~RosFcnnInference()
{
    this->deallocateCudaMemory();
    this->destroyEngine();
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
        this->grabImageAndPreprocess();
        this->process();
        this->publishOutput();
        ROS_INFO( "OK!" );
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

    float meanR, meanG, meanB;

    this->mNodeHandle.param<float>( "mean_r", meanR, DEFAULT_MEAN_R );
    this->mNodeHandle.param<float>( "mean_g", meanG, DEFAULT_MEAN_G );
    this->mNodeHandle.param<float>( "mean_b", meanG, DEFAULT_MEAN_B );

    this->mMean = make_float3( meanR, meanG, meanB );
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

void RosFcnnInference::setOutputCameraInfo()
{
    std::string imageCameraName;
    if( !camera_calibration_parsers::readCalibration( ros::package::getPath( PACKAGE_NAME ) + "/calib/" + this->mCalibName
                                               , imageCameraName
                                               , this->mImageCameraInfo ) )
    {
        ROS_ERROR( "Cant read calib file" );
    }

    this->mImageCameraInfo.header           = std_msgs::Header();
    this->mImageCameraInfo.header.frame_id  = this->mCameraLink;

    this->mDepthCameraInfo                 = this->mImageCameraInfo;
    this->mDepthCameraInfo.header          = std_msgs::Header();
    this->mDepthCameraInfo.header.frame_id = this->mDepthLink;
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
    
    this->mInputSizeRGB8    = DIMS_C( inputDims ) * DIMS_H( inputDims ) * DIMS_W( inputDims ) * sizeof(uint8_t);
    
    // Get output size
    const int       outputIndex  = this->mCudaEngine->getBindingIndex( this->mEngineOutputName.c_str() );
    nvinfer1::Dims  outputDims   = this->mCudaEngine->getBindingDimensions( outputIndex );
    
    ROS_INFO( "%d %d", sizeof(uchar3), sizeof(uint8_t) );

    this->mOutputSize      = DIMS_C( outputDims ) * DIMS_H( outputDims ) * DIMS_W( outputDims ) * sizeof(float);

    ROS_INFO( "DONE!" );
}

void RosFcnnInference::allocateCudaMemory()
{
    if( CUDA_FAILED( cudaStreamCreateWithFlags( &mCudaStream, cudaStreamDefault ) ) )
    {
        ROS_ERROR( "Failed to create CUDA stream" );
    }

    if( !cudaAllocMapped( (void**)&mImageCPU, (void**)&mImageCUDA, this->mInputSize ) )
    {
        ROS_ERROR( "Cant allocate CUDA memory for mImageCPU" );
    }
    if( !cudaAllocMapped( (void**)&mImageRGB8CPU, (void**)&mImageRGB8CUDA, this->mInputSizeRGB8 ) )
    {
        ROS_ERROR( "Cant allocate CUDA memory for mImageRGB8CPU" );
    }
    if( !cudaAllocMapped( (void**)&mImageRGBACPU, (void**)&mImageRGBACUDA, this->mInputImageHeight * this->mInputImageWidth * sizeof( float4 ) ) )
    {
        ROS_ERROR( "Cant allocate CUDA memory for mImageRGBA" );
    }
    if( !cudaAllocMapped( (void**)&mOutImageCPU, (void**)&mOutImageCUDA, this->mOutputSize ) )
    {
        ROS_ERROR( "Cant allocate CUDA memory for mOutImageCPU" );
    }
    
}

void RosFcnnInference::deallocateCudaMemory()
{
    cudaFreeHost( mOutImageCPU );
    cudaFree( mOutImageCUDA );
    
    cudaFreeHost( mImageRGBACPU );
    cudaFree( mImageRGBACUDA );

    cudaFreeHost( mImageRGB8CPU );
    cudaFree( mImageRGB8CUDA );

    cudaFreeHost( mImageCPU );
    cudaFree( mImageCUDA );

    cudaStreamDestroy( mCudaStream );
}

void RosFcnnInference::destroyEngine()
{
    this->mExecutionContext->destroy();
    this->mCudaEngine->destroy();
    this->mRuntimeInfer->destroy();
}

void RosFcnnInference::destroyCamera()
{
    this->mCamera->Close();
}

void RosFcnnInference::grabImageAndPreprocess()
{
    if( this->mUseInternalCamera )
    {
        if( this->mCamera->Capture( &this->mImageCPU, &this->mImageCUDA, 1000 ) )
        {
            ROS_WARN( "Failed to capture frame" );
        }

        if( this->mCamera->ConvertRGBA( this->mImageCUDA, &this->mImageRGBACPU, true ) )
        {
            ROS_WARN( "Failed to convert from NV12 to RGBA" );
        }

        CUDA( cudaRGBA32ToRGB8( (float4*) this->mImageRGBACPU, (uchar3*) this->mImageRGB8CUDA
                               , this->mInputImageWidth, this->mInputImageHeight ) );
    }
    else
    {
        this->mRosImageMsg = ros::topic::waitForMessage<sensor_msgs::Image>( "/image" );
        this->mCvImage     = cv_bridge::toCvCopy( this->mRosImageMsg );
        cv::cvtColor( this->mCvImage->image, this->mCvImage->image, cv::COLOR_BGR2RGB );

        cudaMemcpy2D( mImageRGB8CUDA, this->mInputImageWidth * sizeof(uchar3)
                    , (void*)mCvImage->image.data, this->mCvImage->image.step
                    , this->mInputImageWidth * sizeof(uchar3), this->mInputImageHeight, cudaMemcpyHostToDevice );

        cudaRGB8ToRGBA32( (uchar3*)mImageRGB8CUDA, (float4*)mImageRGBACUDA, this->mInputImageWidth, this->mInputImageHeight );

    }
    
    if( CUDA_FAILED( cudaPreImageNetMean( (float4*)mImageRGBACUDA, mInputImageWidth, mInputImageHeight
                                        , (float*)mImageRGB8CUDA, mInputImageWidth, mInputImageHeight
                                        , this->mMean, this->mCudaStream ) ) )
    {
        ROS_ERROR( "Failed to preprocess" );
    }

    ROS_INFO( "Image preprocessed" );
}

void RosFcnnInference::process()
{
    void* bindings[] = {mImageRGB8CUDA, mOutImageCUDA};
    this->mExecutionContext->execute( 1, bindings );
}

void RosFcnnInference::publishOutput()
{
    this->mOutRosImageMsg = cv_bridge::CvImage( std_msgs::Header()
                                              , "rgb8"
                                              , this->mCvImage->image 
                                              ).toImageMsg();
    
    cv::Mat outDepth( this->mInputImageHeight, this->mInputImageWidth, CV_32FC1, this->mOutImageCUDA );
    this->mOutRosDepthMsg = cv_bridge::CvImage( std_msgs::Header()
                                              , sensor_msgs::image_encodings::TYPE_32FC1
                                              , outDepth
                                              ).toImageMsg();

    ros::Time timestamp = ros::Time::now();

    this->mOutRosImageMsg->header.stamp = timestamp;
    this->mOutRosDepthMsg->header.stamp = timestamp;
    
    this->mImageCameraInfo.header.stamp = timestamp;
    this->mDepthCameraInfo.header.stamp = timestamp;

    this->mImageInfoPublisher.publish( this->mImageCameraInfo );
    this->mImagePublisher.publish( this->mOutRosImageMsg );
    
    this->mDepthInfoPublisher.publish( this->mDepthCameraInfo );
    this->mDepthPublisher.publish( this->mOutRosDepthMsg );
    

}