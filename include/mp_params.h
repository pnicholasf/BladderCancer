//
// Created by franklin on 6/24/26.
//

#ifndef MP_PARAMS_H
#define MP_PARAMS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

class mp_params
{
public:
  explicit mp_params(std::string configfilepath);
  ~mp_params()=default;
  void parse_yaml();
  std::string configfilepath;
  cv::Size frame_size, frame_step, patch_size, patch_step;
  bool cb_grid_wsi{}, cb_grid_patch{};
  float cb_samp_wsi{}, cb_samp_patch{};
};

#endif //MP_PARAMS_H
