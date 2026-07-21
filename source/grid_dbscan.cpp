//
// Created by franklin on 7/10/26.
//

#include "grid_dbscan.h"

/*....................................................................................................................*/
GridDbClust::GridDbClust(
    const float epsilon, const int32_t minPts, const int32_t radius,
    const int32_t min_clust_size, const cv::NormTypes norm_type
    )
{
    this->epsilon = epsilon;
    this->minPts = minPts;
    this->radius = radius;
    this->min_clust_size = min_clust_size;
    this->norm_type = norm_type;
}

/*....................................................................................................................*/
void GridDbClust::range_query(const int32_t row, const int32_t col, std::vector<cv::Point> &points) const
{
    const int32_t row_min = std::max(0, row - this->radius);
    const int32_t row_max = std::min(this->size.height, row + this->radius);
    const int32_t col_min = std::max(0, col - this->radius);
    const int32_t col_max = std::min(this->size.width, col + this->radius);
    points.clear();

    const auto rng1 = std::vector<cv::Range>{
        cv::Range::all(), cv::Range(row, row + 1), cv::Range(col, col + 1)
    };
    auto v1 = this->idata(rng1).reshape(1);

    for (int32_t rr = row_min; rr <= row_max; rr++)
    {
        for (int32_t cc = col_min; cc <= col_max; cc++)
        {
            if (rr==row && cc == col) continue;
            auto rng2 = std::vector<cv::Range>{
                cv::Range::all(), cv::Range(rr, rr + 1), cv::Range(cc, cc + 1)
            };
            auto v2 = this->idata(rng2).reshape(1);

            double dd = cv::norm(v1, v2);
            if (dd <= this->epsilon)
                points.emplace_back(cc, rr);
        }
    }
}

/*....................................................................................................................*/
void GridDbClust::filter_cluster(cv::Mat &label_map) const
{
    std::vector<int32_t> uvals, counts;
    unique_values<int32_t>(label_map, uvals);
    count_instances(label_map, uvals, counts);
    for (size_t ii = 0; ii < counts.size(); ii++)
    {
        if (counts[ii] >= this->min_clust_size) continue;
        cv::Mat indx = label_map == uvals[ii];
        label_map.setTo(0, indx);
    }

    cv::Mat olab = cv::Mat::zeros(label_map.rows, label_map.cols, CV_32S);
    unique_values<int32_t>(label_map, uvals);
    int32_t lcnt=1;
    for (auto uval : uvals)
    {
        if (uval==0 || uval == -1) continue;
        auto indx = label_map == uval;
        olab.setTo(lcnt, indx);
        lcnt++;
    }
    label_map = olab;
}

/*....................................................................................................................*/
void GridDbClust::get_clusters(cv::Mat& feature_data, cv::Mat& mask, cv::Mat& label)
{
    this->idata = feature_data;
    std::cout<<this->idata.size[0]<<" "<<this->idata.size[1]<<" "<<this->idata.size[2]<<std::endl;
    this->size = cv::Size(feature_data.size[1], feature_data.size[2]);
    this->features = feature_data.size[0];
    if (mask.empty())
        mask = cv::Mat::ones(this->size.height, this->size.width, CV_32S);
    label = cv::Mat::zeros(this->size.height, this->size.width, CV_32S);

    int32_t ccount = 0;
    const size_t frame_step = this->size.height * this->size.width;
    const auto *p_mask = reinterpret_cast<int32_t*>(mask.data);
    auto *p_label = reinterpret_cast<int32_t*>(label.data);

    for (int32_t r = 0; r < this->size.height; r++)
        for (int32_t c = 0; c < this->size.width; c++)
        {
            size_t l_ind = r * this->size.width + c;
            if (p_mask[l_ind]==0 || p_label[l_ind]!=0) continue;

            std::vector<cv::Point> seed_set;
            this->range_query(r, c, seed_set);
            if (seed_set.size() < this->minPts)
            {
                p_label[l_ind] = -1;
                continue;
            }
            ccount += 1;
            p_label[l_ind] = ccount;
            //std::vector<cv::Point> seed_set(nbrs) ;
            size_t l_count = 0;
            while (l_count < seed_set.size())
            {
                const int32_t c1 = seed_set[l_count].x;
                const int32_t r1 = seed_set[l_count].y;
                l_count++;
                l_ind = r1 * this->size.width + c1;
                if (p_mask[l_ind] == 0) continue;
                if (p_label[l_ind] == -1)
                {
                    p_label[l_ind] = ccount;
                    continue;
                }
                if (p_label[l_ind] !=0) continue;
                p_label[l_ind] = ccount;
                std::vector<cv::Point> nbrs1;
                this->range_query(r1, c1, nbrs1);
                if (nbrs1.size() >= this->minPts)
                    for (cv::Point p : nbrs1)
                        seed_set.push_back(p);
            }
        }
    if (this->min_clust_size > 0)
        this->filter_cluster(label);
    this->idata.release();
}

/*....................................................................................................................*/
