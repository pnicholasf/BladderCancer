//
// Created by franklin on 7/10/26.
//

#ifndef GRID_DBSCAN_H
#define GRID_DBSCAN_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "gridutils.h"

class GridDbClust
{
public:
  explicit GridDbClust(
    float epsilon, int32_t minPts=1, int32_t radius=1,
    int32_t min_clust_size=-1, cv::NormTypes norm_type=cv::NORM_L2
    );
  void get_clusters(cv::Mat & feature_data, cv::Mat & mask, cv::Mat & label);
  void filter_cluster(cv::Mat& label_map) const;
  ~GridDbClust()=default;

private:
  cv::Mat idata;
  cv::Size2i size;
  int32_t features=-1;
  float epsilon;
  int32_t minPts, radius{}, min_clust_size{};
  cv::NormTypes norm_type;
  void range_query(int32_t row, int32_t col, std::vector<cv::Point> & points) const;
};

#endif //GRID_DBSCAN_H
