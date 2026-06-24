//
// Created by franklin on 6/24/26.
//
#include "mp_params.h"
 mp_params::mp_params(std::string configfilepath) {
   this->configfilepath = std::move(configfilepath);
   this->parse_yaml();
 }

void mp_params::parse_yaml() {
   YAML::Node config = YAML::LoadFile(this->configfilepath);
   //this->frame_size.clear();
   std::vector<int32_t> sz{};

   YAML::Node tmp = config["frame_size"];

   sz.clear();
   for(YAML::const_iterator it=tmp.begin(); it != tmp.end(); ++it)
     sz.emplace_back(it->as<int32_t>());
   this->frame_size = cv::Size(sz[0], sz[1]);

   tmp = config["frame_step"];
   sz.clear();
   for(YAML::const_iterator it=tmp.begin(); it != tmp.end(); ++it)
     sz.emplace_back(it->as<int32_t>());
   this->frame_step = cv::Size(sz[0], sz[1]);

   tmp = config["patch_size"];
   sz.clear();
   for(YAML::const_iterator it=tmp.begin(); it != tmp.end(); ++it)
     sz.emplace_back(it->as<int32_t>());
   this->patch_size = cv::Size(sz[0], sz[1]);

   tmp = config["patch_step"];
   sz.clear();
   for(YAML::const_iterator it=tmp.begin(); it != tmp.end(); ++it)
     sz.emplace_back(it->as<int32_t>());
   this->patch_step = cv::Size(sz[0], sz[1]);

   this->cb_grid_wsi = config["cb_grid_wsi"].as<bool>();
   this->cb_grid_patch = config["cb_grid_patch"].as<bool>();
   this->cb_samp_wsi = config["cb_samp_wsi"].as<float>();
   this->cb_samp_patch = config["cb_samp_patch"].as<float>();
 }