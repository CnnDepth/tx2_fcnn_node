# tx2_fcnn_node

ROS node for real-time FCNN-based depth reconstruction (as in paper TODO: add url). The platforms are NVidia Jetson TX2 and x86_64 PC with GNU/Linux (aarch64 should work as well, but not tested).

## Publications

If you use this work in an academic context, please cite the following publication(s):

TBD

## System requirements

* Linux-based system with aarch64 or x86_64 architecture or NVidia Jetson TX2.
* NVidia graphic card.

## Pre-requesites

1) ROS Kinetic or higher.
2) TensorRT 5.0 or higher.
3) CUDA 9.0 or higher
4) CUDNN + CuBLAS
5) GStreamer-1.0
6) glib2.0

*Optional:*
* RTAB-MAP

## Compile

Assuming you already have ROS and CUDA related tools installed

1) Install remaining pre-requesites:

```bash
$ sudo apt-get update
$ sudo apt-get install -y libqt4-dev qt4-dev-tools \ 
       libglew-dev glew-utils libgstreamer1.0-dev \ 
       libgstreamer-plugins-base1.0-dev libglib2.0-dev \
       libgstreamer-plugins-good
$ sudo apt-get install -y libopencv-calib3d-dev libopencv-dev 

```

2) Navigate to your catkin workspace and clone the repository:

```bash
$ git clone https://github.com/CnnDepth/tx2_fcnn_node.git
$ cd tx2_fcnn_node && git submodule update --init --recursive
```

3) Build the node:

Navigate to catkin workspace folder.

**On jetson:**

```console 
$ catkin_make
```

**On x86_64 PC**
```bash
$ catkin_make --cmake-args -DPATH_TO_TENSORRT_LIB=/usr/lib/x86_64-linux-gnu \ 
              -DPATH_TO_TENSORRT_INCLUDE=/usr/include -DPATH_TO_CUDNN=/usr/lib/x86_64-linux-gnu \ 
              -DPATH_TO_CUBLAS=/usr/lib/x86_64-linux-gnu
```

Change the paths accordingly.

4) Run:

```bash
$ roslaunch tx2_fcnn_node cnn_only.launch
```

or with RTAB-MAP

```bash
$ roslaunch tx2_fcnn_node rtabmap_cnn.launch
```
## Run in a container

1) Build image:
```bash
$ cd docker
$ docker build . -t rt-ros-docker
```
2) Run an image:
```bash
$ nvidia-docker run -device=/dev/video0:/dev/video0 -it --rm rt-ros-docker
```
3) Create ros workspace:
```bash
$ mkdir -p catkin_ws/src && cd catkin_ws/src
$ catkin_init_workspace
$ cd ..
$ catkin_make
$ source devel/setup.bash
```
4) Build tx2_fcnn_node:
```bash
$ cd src
$ git clone https://github.com/CnnDepth/tx2_fcnn_node.git
$ cd tx2_fcnn_node && git submodule update --init --recursive
$ catkin_make
```
5) Run the node:
```bash
rosrun tx2_fcnn_node tx2_fcnn_node
```
## Nodes
### tx2_fcnn_node
Reads the images from camera or image topic and computes the depth map.

#### Subscribed Topics
* **`/image`** ([sensor_msgs/Image])
       
     The input color image for depth reconstruction
       
#### Published topics
* **`/rgb/image`** ([sensor_msgs/Image])

     The output color image.
       
* **`/depth/image`** ([sensor_msgs/Image])

    The output depth map. The image is in CV_32FC1.

* **`/rgb/camera_info`** ([sensor_msgs/CameraInfo])
    
    Camera info.

* **`/depth/camera_info`** ([sensor_msgs/CameraInfo])
    
    "Depth" camera info. Duplicates /rgb/camera_info

#### Parameters

* **`input_width`** (int, default: 320)
    
    Input image width for TensorRT engine
    
* **`input_height`** (int, default: 240)

    Input image height for TensorRT engine
    
* **`use_camera`** (bool, default: true)
   
    If true - use internal camera as image source. False - use /image topic as input source.
    
* **`camera_mode`** (int, default: -1)

    Only works if use_camera:=true. Sets camera device to be opened. -1 - default device. 
    
* **`camera_link`** (string, default: "camera_link")
  
     Name of camera's frame_id.
     
* **`depth_link`** (string, default: "depth_link")

    Name of depth's frame_id
    
* **`engine_name`** (string, default: "test_engine.trt")

    Name of the compiled TensorRT engine file, localed in "engine" folder.
    
* **`calib_name`** (string, default: "tx2_camera_calib.yaml")

    Name of calibration file, obrained with camera_calib node. May be either in .yaml or .ini format.
    
* **`input_name`** (string, default: "tf/Placeholder")

    Name of the input of TensorRT engine.
    
* **`output_name`** (string, default: "tf/Reshape")

    Name of the output of TensorRT engine
    
* **`mean_r`** (float, default: 123.0)

    R channel mean value, used during FCNN training.
    
* **`mean_g`** (float, default: 115.0)

    G channel mean value, used during FCNN training.
    
* **`mean_b`** (float, default: 101.0)

    B channel mean value, used during FCNN training.

## Sample models

Models pre-trained on NYU Depth v2 dataset are available in [http://pathplanning.ru/public/ECMR-2019/engines/](http://pathplanning.ru/public/ECMR-2019/engines/). The models are stored in UFF format. They can be converted into TensorRT engines using [tensorrt_samples](https://github.com/CnnDepth/tensorrt_samples/tree/master/sampleUffFCRN).

## Troubleshooting

**Stack smashing**

If you run this node on Ubuntu 16.04 or older, the node may fail to start and show `Stack smashing detected` log message. To fix it, remove `XML.*` files in `Thirdparty/fcrn-inference/jetson-utils` directory, and recompile the project.

**Inverted image**

If you run this node on Jetson, RGB and depth image may be shown inverted. To fix it, open `Thirdparty/fcrn-inference/jetson-utils/camera/gstCamera.cpp` file in text editor, go to lines 344-348, and change value of `flipMethod` constant to 0. After editing, recompile the project.

[sensor_msgs/Image]: http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html
[sensor_msgs/CameraInfo]: http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CameraInfo.html
