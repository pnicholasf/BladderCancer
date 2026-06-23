//
// Created by franklin on 6/17/26.
//

#include "test_utils.h"

// -----------------------------------------------------------------------------
void test_arange()
{
  const std::vector<int> xx = arange<int> (0, 10, 2);
  for (const int x : xx) {
    std::cout<<x<<" ";
  }
  std::cout<<std::endl;
}

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
//*
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
//*/
// -----------------------------------------------------------------------------
void test_wsi_grid()
{
  cv::Size wsi_size(1000,900);
  cv::Size frame_size(250,250);
  cv::Size frame_step(-1,-1);
  cv::Size patch_size(50,50);
  cv::Size patch_step(25,25);
  cv::Mat coords;
  bool cb_grid = true;

  sample_wsi_grid(
    wsi_size, frame_size,
    frame_step, patch_size, patch_step,
    coords, cb_grid, true
    );

  std::cout<<"coords shape("<< coords.rows << ","<<coords.cols<<std::endl;
  for (int ir=0; ir < coords.rows; ir++) {
    //std::cout<<ir<<"  "<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<std::endl;
    std::cout<<coords.at<int>(ir, 0)<<","<<coords.at<int>(ir, 1)<<","<<std::endl;
  }
}

// -----------------------------------------------------------------------------
