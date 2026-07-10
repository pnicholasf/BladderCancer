//
// Created by franklin on 7/10/26.
//

#include "grid_dbscan.h"

/*....................................................................................................................*/
GridDbClust::GridDbClust(
    const float epsilon, const int32_t minPts, const int32_t radius,
    const int32_t min_clust_size, const cv::NormTypes norm_type, bool channel_first
    )
{
    this->epsilon = epsilon;
    this->minPts = minPts;
    this->radius = radius;
    this->min_clust_size = min_clust_size;
    this->norm_type = norm_type;
    this->channel_first = channel_first;
}

/*....................................................................................................................*/
void GridDbClust::range_query(const int32_t row, const int32_t col, std::vector<cv::Point> &points) const
{
    const int32_t row_min = std::max(0, row - this->radius);
    const int32_t row_max = std::min(this->size.height, row + this->radius);
    const int32_t col_min = std::max(0, col - this->radius);
    const int32_t col_max = std::min(this->size.width, col + this->radius);
    points.clear();
    for (int32_t rr = row_min; rr <= row_max; rr++)
        for (int32_t cc = col_min; cc <= col_max; cc++)
        {
            if (rr==cc)
                continue;
            if (this->channel_first)
            {
                std::vector<cv::Range> rng1{cv::Range::all(), cv::Range(row, row+1), cv::Range(col, col+1)};
                std::vector<cv::Range> rng2{cv::Range::all(), cv::Range(rr, rr+1), cv::Range(cc, cc+1)};
            }
            else
            {
                std::vector<cv::Range> rng1{cv::Range(row, row+1), cv::Range(col, col+1), cv::Range::all()};
                std::vector<cv::Range> rng2{cv::Range(rr, rr+1), cv::Range(cc, cc+1), cv::Range::all()};
            }

            double dd = cv::norm(this->idata(rng1), this->idata(rng2), this->norm_type);
            if (dd <= this->epsilon)
                points.emplace_back(cc, rr);
        }
}

/*....................................................................................................................*/
void GridDbClust::get_clusters(const cv::Mat& feature_data, cv::Mat& mask, cv::Mat& label)
{
    this->idata = feature_data;
    this->size = cv::Size(feature_data.cols, feature_data.rows);
    if (mask.empty())
        mask = cv::Mat::ones(this->size.height, this->size.width, CV_32S);
    label = cv::Mat::zeros(this->size.height, this->size.width, CV_32S);

    int32_t ccount = 0;
    for (int32_t r = 0; r < this->size.height; r++)
        for (int32_t c = 0; c < this->size.width; c++)
        {
            size_t l_ind = r * this->size.width + c;
            if (mask.data[l_ind] == 0) continue;
            if (label.data[l_ind] != 0) continue;

            std::vector<cv::Point> nbrs;
            this->range_query(r, c, nbrs);
            if (nbrs.size() < this->minPts)
            {
                label.data[l_ind] = -1;
                continue;
            }
            ccount += 1;
            label.data[l_ind] = ccount;
            std::vector<cv::Point> seed_set(nbrs) ;
            size_t l_count = 0;
            while (l_count < seed_set.size())
            {
                const int32_t c1 = seed_set[l_count].x;
                const int32_t r1 = seed_set[l_count].y;
                l_count++;
                l_ind = r1 * this->size.width + c1;
                if (mask.data[l_ind] == 0) continue;
                if (label.data[l_ind] != -1) label.data[l_ind] = ccount;
                if (label.data[l_ind] !=0) continue;
                label.data[l_ind] = ccount;
                std::vector<cv::Point> nbrs1;
                this->range_query(r, c, nbrs1);
                if (nbrs1.size() < this->minPts)
                    for (cv::Point p : nbrs1)
                        seed_set.push_back(p);
            }
        }
    this->idata.release();
}

/*....................................................................................................................*/
