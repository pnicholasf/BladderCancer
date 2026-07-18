//
// Created by franklin on 6/17/26.
//

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <iostream>
#include <fstream>
#include "gridutils.h"
#include "grid_dbscan.h"
#include "mp_params.h"

void test_arange();
void test_meshgrid();
void test_samplegrid();
void test_wsi_grid();
void test_mp_params();
void test_wsi_grid_pars();
void load_dbclust_data(int32_t nrows, int32_t ncols, int32_t nfeatures, cv::Mat &data, cv::Mat& mask);
void generate_test_data(
    cv::Mat& odata, cv::Mat& mask,
    int32_t rows=128, int32_t cols=128, int32_t features=8,
    bool chan_first=true,
    const std::string& outdir=""
    );
void test_dbclust();

#endif //TEST_UTILS_H
