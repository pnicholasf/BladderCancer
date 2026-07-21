//
// Created by franklin on 7/21/26.
//
#include "postprocess.h"

/*....................................................................................................................*/
void split_cyto_nuc(const cv::Mat& cytonuc_mask, const cyto_range& crange, cv::Mat& cytomask, cv::Mat& nucmask)
{
    // cytoplasm regions
    cytomask.create(cytonuc_mask.rows, cytonuc_mask.cols, CV_32S);
    cytomask = 0;
    cv::Mat indx = (cytonuc_mask >= crange.low) & (cytonuc_mask <= crange.high);
    cytomask.setTo(cytonuc_mask, indx);

    // nucleii regions
    nucmask.create(cytonuc_mask.rows, cytonuc_mask.cols, CV_32S);
    nucmask = 0;
    indx = cytonuc_mask > crange.high;
    nucmask.setTo(cytonuc_mask, indx);
}

/*....................................................................................................................*/
void relate_nuc2cyto(
    cv::Mat& cytomask, cv::Mat& nucmask, float nuc_frac, std::unordered_map<int32_t, std::vector<int32_t>>& cnmap
    )
{
    std::vector<int32_t> id_cyto, id_nuc;
    unique_values<int32_t>(cytomask, id_cyto);
    unique_values<int32_t>(nucmask, id_nuc);
    auto ncyto = static_cast<int32_t>(id_cyto.size());
    auto nnuc = static_cast<int32_t>(id_nuc.size())-1;
    std::vector<bool> id_avail(nnuc, true);

    cnmap.clear();
    for (const auto idc : id_cyto)
    {
        // skip background
        if (idc == 0) continue;

        // get cyto mask
        cv::Mat cyto_comp = cytomask == idc;
        cyto_comp.convertTo(cyto_comp, CV_32S);

        // get convex hull of cell
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(cyto_comp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;
        cnmap[idc] = std::vector<int32_t>();
        std::vector<cv::Point> cc = contours[0];
        for (int ii=1; ii<contours.size(); ii++)
        {
            cc.insert(cc.begin(), contours[ii].begin(), contours[ii].end());
        }
        std::vector<cv::Point> c_hull;
        cv::convexHull(cc, c_hull, true, true);
        c_hull.push_back(c_hull[0]);
        cyto_comp = 0;
        cv::fillConvexPoly(cyto_comp, c_hull, true);

        // find nuclei contained in cyto mask
        for (int32_t ii = 0; ii < id_nuc.size(); ii++)
        {
            if (id_nuc[ii] == 0) continue;
            cv::Mat nuc_comp = nucmask == ii;
            cv::Mat in_cyto = cyto_comp & nuc_comp;
            in_cyto.convertTo(in_cyto, CV_32S);
            nuc_comp.convertTo(nuc_comp, CV_32S);
            if (cv::sum(in_cyto)[0] > nuc_frac * cv::sum(nuc_comp)[0])
            {
                cnmap[idc].push_back(id_nuc[ii]);
                id_avail[ii] = false;
            }
        }
    }

    // mark free nuclei
    for (int32_t ii = 0; ii < id_nuc.size(); ii++)
        if (id_avail[ii]) cnmap[-1].push_back(id_nuc[ii]);
}

/*....................................................................................................................*/
