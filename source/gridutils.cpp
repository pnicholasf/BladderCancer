//
// Created by franklin on 6/17/26.
//

#include "gridutils.h"

/*............................................................................*/
void sample_grid(
  cv::Size &window_size,
  cv::Size &frame_size,
  cv::Size &frame_step,
  cv::Mat &coords,
  bool exclude_border,
  bool cb_grid,
  float cb_samp
  )
{
  auto fsh = frame_step.height;
  auto fsw = frame_step.width;

  if (cb_grid)
  {
    fsh *= static_cast<int>(cb_samp);
    fsw *= static_cast<int>(cb_samp);
  }

  auto sr = arange<int32_t>(0, window_size.height, fsh);
  auto sc = arange<int32_t>(0, window_size.width, fsw);

  std::vector<int32_t> row_rng, col_rng;
  if (exclude_border)
  {
    for (auto rv : sr) {
      if (rv + fsh < window_size.height)
        row_rng.push_back(rv);
    }
    for (auto cv : sc){
      if (cv + fsw < window_size.width)
        col_rng.push_back(cv);
    }
  }
  else
  {
    row_rng = sr;
    col_rng = sc;
  }
  //cv::Mat_<int32_t> coords1;
  cv::Mat_<int32_t> row_coords, col_coords;
  meshgrid<int32_t>(row_rng, col_rng, row_coords, col_coords);
  cv::Mat rr, cc;
  if (cb_grid){
    int hfsw = fsw/2;
    for (int ii=1; ii < row_coords.rows ; ii += 2) {
      col_coords.rowRange(ii, ii+1) += hfsw;
    }
    cv::Mat indx = col_coords + frame_size.width < window_size.width;
    cv::copyTo(row_coords, rr, indx);
    cv::copyTo(col_coords, cc, indx);
  }
  else{
    rr = row_coords;
    cc = row_coords;
  }

  auto nelems  = static_cast<int>(rr.total());
  rr = rr.reshape(1, nelems);
  cc = cc.reshape(1, nelems);
  cv::hconcat(cc, rr, coords);
}

/*............................................................................*/
