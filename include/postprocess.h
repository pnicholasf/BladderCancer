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

//

typedef std::unordered_map<int32_t, std::vector<int32_t>> CELLMAP;
typedef std::unordered_map<int32_t, double> NCRATIO;

struct cyto_range
{
    int32_t low=1;
    int32_t high=1000;
};

struct vrange
{
    float low;
    float high;
};
typedef std::unordered_map<std::string, vrange> MORPH_CONSTR;

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
    CELLMAP& cnmap //cyto-nuc map
    );

void filter_singles(
    const cv::Mat& cytonuc_mask,
    cv::Mat& omask,
    CELLMAP& map_in,
    CELLMAP& map_out,
    float nuc_frac=DEFAULT_NUC_FRAC,
    const cyto_range& crange
    );

void get_nc_ratio(
    cv::Mat& cytomask,
    cv::Mat& nucmask,
    NCRATIO& ncratio,
    float nuc_frac=DEFAULT_NUC_FRAC
    );

class FilterMask
{
public:
    explicit FilterMask(
        const MORPH_CONSTR &cyto_constr, const MORPH_CONSTR &nuc_constr,
        bool exclude_bare_cyto=true, bool exclude_bare_nuc=true
        );
    bool test_if_valid(const cv::Mat& image, const cv::Mat& mask, int32_t id2test, const std::string& constr_type);
    void filter(
        const cv::Mat& image, const cv::Mat& cytomask_i, const cv::Mat& nucmask_i,
        cv::Mat& cytomask_o, cv::Mat& nucmask_o, CELLMAP& map_in, CELLMAP& map_out
    );

private:
    MORPH_CONSTR cyto_constr, nuc_constr;
    bool exclude_bare_cyto, exclude_bare_nuc;
    int32_t test(double val, vrange rng);
};

#endif //POSTPROC_H
