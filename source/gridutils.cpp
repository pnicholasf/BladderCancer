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

  //############################################################################
  /*
  std::cout<<"row coords"<<std::endl;
  for (int ir=0; ir < row_coords.rows; ir++) {
    for (int ic=0; ic < row_coords.cols; ic++) {
      std::cout<<row_coords(ir, ic)<< "  ";
    }
    std::cout<<std::endl;
  }
  std::cout<<"col coords"<<std::endl;
  for (int ir=0; ir < col_coords.rows; ir++) {
    for (int ic=0; ic < col_coords.cols; ic++) {
      std::cout<<col_coords(ir, ic)<< "  ";
    }
    std::cout<<std::endl;
  }
  */
  //############################################################################

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
    cc = col_coords;
  }

  int nelems  = static_cast<int>(rr.total());
  coords.create(nelems, 2, CV_32S);
  auto d_ptr = reinterpret_cast<int32_t *>(coords.data);
  auto r_ptr = reinterpret_cast<int32_t *>(rr.data);
  auto c_ptr = reinterpret_cast<int32_t *>(cc.data);

  for (int ii=0; ii<nelems; ii++) {
    *d_ptr++ = *c_ptr++; // col -> X coord
    *d_ptr++ = *r_ptr++; // row -> Y coord
  }
}

/*............................................................................*/
void sample_grid_wsi(
  cv::Size &wsi_size, cv::Size &frame_size, cv::Size &frame_step,
  cv::Size &patch_size, cv::Size &patch_step, cv::Mat& coords,
  const bool cb_grid, const float cb_samp
  )
{
  if (frame_step == cv::Size(-1,-1))
    frame_step = cv::Size(frame_size.width-patch_size.width,frame_size.height-patch_size.height);

  cv::Mat frame_coords, patch_coords;
  sample_grid(
    wsi_size, frame_size, frame_step, frame_coords,
    false
    );
  sample_grid(
    frame_size, patch_size, patch_step,patch_coords,
    true, cb_grid, cb_samp
    );

  std::vector<int32_t>  vX, vY;
  for (int ii=0; ii< frame_coords.rows; ii++)
    {
      for (int jj=0; jj<patch_coords.rows; jj++)
        {
        int32_t xx = frame_coords.at<int32_t>(ii, 0);
        int32_t yy = patch_coords.at<int32_t>(jj, 1);
        if (xx+patch_size.width < wsi_size.width && yy+patch_size.height < wsi_size.height) {
          vX.push_back(xx);
          vY.push_back(yy);
        }
    }
  }
  coords.create(vX.size(),2, CV_32S);
  auto d_ptr = reinterpret_cast<int32_t *>(coords.data);
  for (int ii=0; ii<vX.size(); ii++) {
    *d_ptr++ = vX[ii];
    *d_ptr++ = vY[ii];
  }
}
/*............................................................................*/

