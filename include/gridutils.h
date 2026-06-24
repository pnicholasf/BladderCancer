//
// Created by franklin on 6/17/26.
//

#ifndef GRIDUTILS_H
#define GRIDUTILS_H

#include <opencv2/opencv.hpp>
#include <vector>

// -----------------------------------------------------------------------------
void sample_grid(
    cv::Size &window_size, cv::Size &frame_size, cv::Size &frame_step,
    cv::Mat &coords, bool exclude_border=false, bool cb_grid=false,
    float cb_samp = 2.0
    );

// -----------------------------------------------------------------------------
void sample_wsi_grid(
    cv::Size &wsi_size, cv::Size &frame_size, cv::Size &frame_step,
    cv::Size &patch_size, cv::Size &patch_step, cv::Mat& coords,
    bool cb_grid_wsi=false, float cb_samp_wsi = 2.0,
    bool cb_grid_patch=false, float cb_samp_patch = 2.0
    );

// -----------------------------------------------------------------------------
template <typename T>
std::vector<T> arange(T start, T stop, T step=1) {
    std::vector<T> result;
    for (T i = start; i < stop; i += step) {result.push_back(i);}
    return result;
}
// -----------------------------------------------------------------------------
template <typename T>
void meshgrid(
    const std::vector<T>& row_range,
    const std::vector<T>& col_range,
    cv::Mat_<T>& row_coords,
    cv::Mat_<T>& col_coords
    )
{
    auto nrows = row_range.size();
    auto ncols = col_range.size();
    auto nelems = nrows * ncols;
    row_coords.create(nrows, ncols);
    col_coords.create(nrows, ncols);
    T* p_row = reinterpret_cast<T*>(row_coords.data); //(T*) row_coords.data;
    T* p_col = reinterpret_cast<T*>(col_coords.data); //(T*) col_coords.data;
    for(T rr : row_range) {
        std::fill_n(p_row, ncols, rr);
        p_row += ncols;
        std::copy(col_range.begin(), col_range.end(), p_col);
        p_col += ncols;
    }
}

// -----------------------------------------------------------------------------

#endif //GRIDUTILS_H
