//
// Created by franklin on 6/17/26.
//

#include "test_utils.h"

#include "mp_params.h"

// ---------------------------------------------------------------------------------------------------------------------
void test_arange()
{
  const std::vector<int> xx = arange<int> (0, 10, 2);
  for (const int x : xx) {
    std::cout<<x<<" ";
  }
  std::cout<<std::endl;
}

// ---------------------------------------------------------------------------------------------------------------------
void test_meshgrid()
{
  const std::vector<int> rr = arange<int> (0, 10, 2);
  const std::vector<int> cc = arange<int> (0, 20, 2);
  cv::Mat_<int> r_coords, c_coords;
  meshgrid<int>(rr, cc, r_coords, c_coords);
  std::cout<<"row coords"<<std::endl;
  for (int ir=0; ir < r_coords.rows; ir++) {
    for (int ic=0; ic < c_coords.cols; ic++) {
      std::cout<<r_coords(ir, ic)<< "  ";
    }
    std::cout<<std::endl;
  }
  std::cout<<"col coords"<<std::endl;
  for (int ir=0; ir < r_coords.rows; ir++) {
    for (int ic=0; ic < c_coords.cols; ic++) {
      std::cout<<c_coords(ir, ic)<< "  ";
    }
    std::cout<<std::endl;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
void test_samplegrid()
{
  cv::Size win_width(100, 80);
  cv::Size frame_width(10, 10);
  cv::Size frame_step(5, 5);
  cv::Mat coords;
  sample_grid(win_width, frame_width, frame_step, coords, false, true);
  std::cout<<"coords shape("<< coords.rows << ","<<coords.cols<<std::endl;
  for (int ir=0; ir < coords.rows; ir++) {
    //std::cout<<ir<<"  "<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<std::endl;
    std::cout<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<","<<std::endl;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
void test_wsi_grid()
{
  cv::Size wsi_size(5000,5000);
  cv::Size frame_size(1000,1000);
  cv::Size frame_step(-1,-1);
  cv::Size patch_size(64,64);
  cv::Size patch_step(32,32);
  cv::Mat coords;
  constexpr bool cb_grid_wsi = true;
  constexpr float cb_samp_wsi = 2.0;
  constexpr bool cb_grid_patch = true;
  constexpr float cb_samp_patch = 2.0;

  sample_wsi_grid(
    wsi_size, frame_size,
    frame_step, patch_size, patch_step,
    coords, cb_grid_wsi, cb_samp_wsi,
    cb_grid_patch, cb_samp_patch
    );

  for (int ir=0; ir < coords.rows; ir += 1) {
    //std::cout<<ir<<"  "<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<std::endl;
    //std::cout<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<","<<std::endl;
    std::cout<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<std::endl;
  }

  std::cout<<"coords shape("<< coords.rows << ","<<coords.cols<<std::endl;
}

// ---------------------------------------------------------------------------------------------------------------------
void test_mp_params() {
  const std::string configfilepath =
      "/home/franklin/code/cpp/BladderCancer/resources/mp_params.yaml";
  const auto pp = mp_params(configfilepath);
  std::cout<<"mp_params"<<std::endl;
  std::cout<<"frame size: "<<pp.frame_size.width<<", "<<pp.frame_size.height<<std::endl;
  std::cout<<"frame step: "<<pp.frame_step.width<<", "<<pp.frame_step.height<<std::endl;
  std::cout<<"patch size: "<<pp.patch_size.width<<", "<<pp.patch_size.height<<std::endl;
  std::cout<<"patch step: "<<pp.patch_step.width<<", "<<pp.patch_step.height<<std::endl;
  std::cout<<"cb_grid_wsi: "<<pp.cb_grid_wsi<<", "<<"cb_samp_wsi: "<<pp.cb_samp_wsi<<std::endl;
  std::cout<<"cb_grid_patch: "<<pp.cb_grid_patch<<", "<<"cb_samp_patch: "<<pp.cb_samp_wsi<<std::endl;
}

// ---------------------------------------------------------------------------------------------------------------------
void test_wsi_grid_pars() {
  cv::Size wsi_size(5000,5000);
  const std::string configfilepath =
      "/home/franklin/code/cpp/BladderCancer/resources/mp_params.yaml";
  auto pp = mp_params(configfilepath);
  cv::Mat coords;
  sample_wsi_grid(
    wsi_size, pp.frame_size, pp.frame_step, pp.patch_size, pp.patch_step,
    coords, pp.cb_grid_wsi, pp.cb_samp_wsi,
    pp.cb_grid_patch, pp.cb_samp_patch
    );

  for (int ir=0; ir < coords.rows; ir += 1) {
    //std::cout<<ir<<"  "<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<std::endl;
    //std::cout<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<","<<std::endl;
    std::cout<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<std::endl;
  }

  std::cout<<"coords shape("<< coords.rows << ","<<coords.cols<<std::endl;
}

// ---------------------------------------------------------------------------------------------------------------------
void generate_test_data(cv::Mat& odata, cv::Mat& mask, int32_t rows, int32_t cols, int32_t features)
{
  std::vector<cv::Point2d> vcpts{cv::Point2d(30,30), cv::Point2d(80,80)};
  std::vector<int32_t> vrad{10,20};
  const std::vector<int32_t> rrng = arange<int32_t>(0, rows);
  const std::vector<int32_t> crng = arange<int32_t>(0, cols);
  cv::Mat_<int32_t> rr, cc;
  meshgrid(rrng, crng, rr, cc);
  mask = cv::Mat::zeros(rows, cols, CV_32S);
  cv::Mat ones = cv::Mat::ones(rows, cols, CV_32S);

  int sz[3]{features, rows, cols};
  odata = cv::Mat::zeros(3, sz, CV_32F);

  cv::RNG rng;
  for (size_t ii=0; ii < vrad.size(); ii += 1)
  {
    cv::Mat dr = rr - vcpts[ii].y;
    dr = dr.mul(dr);
    cv::Mat dc = cc - vcpts[ii].x;
    dc = dc.mul(dc);
    cv::Mat rad;
    cv::sqrt(dr+dc, rad);
    for (int32_t jj=0; jj < rows; jj += 1)
      for (int32_t kk=0; kk < cols; kk += 1)
        if (rad.at<int32_t>(jj, kk) > vrad[ii]) continue;
        for (int32_t ll=0; ll < features; ll += 1)
        {
          mask.at<int32_t>(ll, jj) = 1;
          odata.at<float>(ll, jj, kk) = static_cast<float>(rng.uniform(0.0, 1.0));
        }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
