//
// Created by franklin on 7/21/26.
//

#ifndef POSTPROC_H
#define POSTPROC_H

#define DEFAULT_NUC_FRAC 0.95

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "gridutils.h"

struct cyto_range
{
    int32_t low=1;
    int32_t high=1000;
};

void split_cyto_nuc(
    const cv::Mat& cytonuc_mask,
    const cyto_range& crange,
    cv::Mat& cytomask,
    cv::Mat& nucmask
    );

void relate_nuc2cyto(
    cv::Mat& cytomask,
    cv::Mat& nucmask,
    float nuc_frac,
    std::unordered_map<int32_t, std::vector<int32_t>>& cnmap //cyto-nuc map
    );

void filter_singles(
    cv::Mat& cytonuc_mask,
    cv::Mat& omask,
    std::unordered_map<int32_t, int32_t>& cnmap,
    float nuc_frac=DEFAULT_NUC_FRAC,
    cyto_range& crange
    );

void get_nc_ratio(
    cv::Mat& cytomask,
    cv::Mat& nucmask,
    float nuc_frac=DEFAULT_NUC_FRAC,
    int32_t radius=-1
    );

#endif //POSTPROC_H
