#ifndef RS_FCNN_NFRNC_HPP
#define RS_FCNN_NFRNC_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <gstCamera.h>
#include <NvInfer.h>
#include <cudaUtility.h>

#include <memory>
#include <string>

class RosFcnnInference
{
    public:
        RosFcnnInference() = default;
        RosFcnnInference( ros::NodeHandle& _nh );
        ~RosFcnnInference();

        void                   run();

    private:
        ros::NodeHandle                    mNodeHandle;
        image_transport::ImageTransport    mImageTransport;

        ros::Subscriber                    mImageSubscriber;
        
        image_transport::Publisher         mImagePublisher;
        image_transport::Publisher         mDepthPublisher;
        ros::Publisher                     mImageInfoPublisher;
        ros::Publisher                     mDepthInfoPublisher;

        // Camera 
        gstCamera*                         mCamera;   

        // Engine related staff
        nvinfer1::IRuntime*                mRuntimeInfer;
        nvinfer1::ICudaEngine*             mCudaEngine;
        nvinfer1::IExecutionContext*       mExecutionContext;

        cudaStream_t                       mCudaStream;
        std::size_t                        mInputSize;
        std::size_t                        mInputSizeUint8;
        std::size_t                        mOutputSize;
        
        std::vector<unsigned char>         mEngineBuffer;

        // ROS parameters list
        int                    mInputImageWidth;
        int                    mInputImageHeight;

        bool                   mUseInternalCamera;
        int                    mCameraMode;

        std::string            mCameraLink;
        std::string            mDepthLink;

        std::string            mEngineName;
        std::string            mCalibName;

        std::string            mEngineInputName;
        std::string            mEngineOutputName;

        float3                 mMean;

        // Images
        void*                  mImageCPU;
        void*                  mImageCUDA;
        void*                  mImageRGBACPU; 
        void*                  mImageRGBACUDA;
        void*                  mImageRGB8CPU;
        void*                  mImageRGB8CUDA;

        float*                 mOutImageCPU;
        float*                 mOutImageCUDA;

        sensor_msgs::ImageConstPtr     mRosImageMsg;
        cv_bridge::CvImagePtr          mCvImage;

        sensor_msgs::ImagePtr          mOutRosImageMsg;
        sensor_msgs::ImagePtr          mOutRosDepthMsg;

        // Private methods
        // Initializers
        void                   initializeParameters();
        void                   initializePublishers();
        void                   initializeInputSource();
        void                   initializeEngine();
        void                   allocateCudaMemory();
        // Deinilializers
        void                   deallocateCudaMemory();
        void                   destroyEngine();
        void                   destroyCamera();

        void                   grabImageAndPreprocess();
        void                   process();
        void                   publishOutput();

};

#endif