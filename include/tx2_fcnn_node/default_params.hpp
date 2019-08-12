#ifndef DFLT_PRMS_HPP
#define DFLT_PRMS_HPP

#include <string>

const std::string      PACKAGE_NAME          = "tx2_fcnn_node";

const int              DEFAULT_INPUT_WIDTH   = 320;
const int              DEFAULT_INPUT_HEIGHT  = 240;

const bool             DEFAULT_USE_CAMERA    = true;
const int              DEFAULT_CAMERA_MODE   = -1;

const std::string      DEFAULT_CAMERA_LINK   = "camera_link";
const std::string      DEFAULT_DEPTH_LINK    = "base_scan";

const std::string      DEFAULT_ENGINE_NAME   = "test_engine.trt";
const std::string      DEFAULT_CALIB_NAME    = "tx2_camera_calib.yaml";

const std::string      DEFAULT_INPUT_NAME    = "tf/Placeholder";
const std::string      DEFAULT_OUTPUT_NAME   = "tf/Reshape";

const float            DEFAULT_MEAN_R        = 123.0;
const float            DEFAULT_MEAN_G        = 115.0;
const float            DEFAULT_MEAN_B        = 101.0;

#endif