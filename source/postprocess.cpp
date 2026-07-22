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
    cv::Mat& cytomask, cv::Mat& nucmask, float nuc_frac, CELLMAP& cnmap
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
            cc.insert(cc.end(), contours[ii].begin(), contours[ii].end());
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
void filter_singles(
    const cv::Mat& cytonuc_mask, cv::Mat& omask,
    CELLMAP& map_in, CELLMAP& map_out,
    const float nuc_frac, const cyto_range& crange
    )
{
    cv::Mat cytomask, nucmask;
    split_cyto_nuc(cytonuc_mask, crange, cytomask, nucmask);
    relate_nuc2cyto(cytomask, nucmask, nuc_frac, map_in);

    map_out = map_in;
    omask = cytonuc_mask.clone();
    for (const auto& [kk, vv] : map_in)
    {
        if (kk==-1) continue;
        else if (vv.size()==1) map_out.erase(kk);
        else
        {
            cv::Mat indx = omask == kk;
            omask.setTo(0, indx);
            for (const auto id_nuc : vv)
            {
                indx = omask == id_nuc + crange.high;
                omask.setTo(0, indx);
            }
        }
    }
}

/*....................................................................................................................*/
void get_nc_ratio(cv::Mat& cytomask, cv::Mat& nucmask, NCRATIO& ncratio, const float nuc_frac)
{
    CELLMAP cnmap;
    relate_nuc2cyto(cytomask, nucmask, nuc_frac, cnmap);
    for (const auto& [id_c, vid_n] : cnmap)
    {
        if (id_c == -1) continue;
        cv::Mat cmsk = cytomask == id_c;
        cmsk.convertTo(cmsk, CV_32S);
        const auto cyto_area = cv::sum(cmsk)[0];
        double nuc_area = 0.0;
        for (const auto id_nuc : vid_n)
        {
            cv::Mat nmsk = nucmask == id_nuc;
            nmsk.convertTo(nmsk, CV_32S);
            nuc_area += cv::sum(nmsk)[0];
        }
        const double  ncr = cyto_area / nuc_area;
        ncratio[id_c] = ncr;
    }
}

/**********************************************************************************************************************/
/*                                          Class: FilterMask
/**********************************************************************************************************************/
FilterMask::FilterMask(
    const MORPH_CONSTR &cyto_constr, const MORPH_CONSTR &nuc_constr,
    const bool exclude_bare_cyto, const bool exclude_bare_nuc)
{
    this->cyto_constr = cyto_constr;
    this->nuc_constr = nuc_constr;
    this->exclude_bare_cyto = exclude_bare_cyto;
    this->exclude_bare_nuc = exclude_bare_nuc;
}

/*....................................................................................................................*/
int32_t FilterMask::test(const double val, const vrange rng)
{
    const int32_t tval = (val >= rng.low) && (val <= rng.high);
    return tval;
}

/*....................................................................................................................*/
bool FilterMask::test_if_valid(
    const cv::Mat& image, const cv::Mat& mask, const int32_t id2test, const std::string& constr_type
    )
{
    MORPH_CONSTR *constr;
    if (constr_type == "cyto")
        constr = &this->cyto_constr;
    else
        constr = &this->nuc_constr;

    std::vector<int32_t> outcomes;
    const cv::Mat mindx = mask == id2test;
    mindx.convertTo(mindx, CV_8UC1);
    const cv::Mat ones = cv::Mat::ones(image.rows, image.cols, CV_32S);
    for (const auto& [kk, vv] : constr)
    {
        switch (kk)
        {
        case "size":
            cv::Mat indx;
            double sz = cv::sum(ones, mindx)[0];
            outcomes.push_back(this->test(sz, vv));
            break;
        case "value":
            double mv = cv::mean(image, mindx)[0];
            outcomes.push_back(this->test(mv, vv));
            break;
        default:
            outcomes.push_back(true);
            break;
        }
    }
    int32_t ret_val = 0;
    for (const auto& outcome : outcomes) ret_val += outcome;
    return ret_val == outcomes.size();
}

/*....................................................................................................................*/
void FilterMask::filter(
    const cv::Mat& image, const cv::Mat& cytomask_i, const cv::Mat& nucmask_i,
    cv::Mat& cytomask_o, cv::Mat& nucmask_o, CELLMAP& map_in, CELLMAP& map_out
    )
{
    cytomask_o = cv::Mat::zeros(image.rows, image.cols, CV_32S);
    nucmask_o = cv::Mat::zeros(image.rows, image.cols, CV_32S);
    int32_t cyto_count = 1;
    int32_t nuc_count = 1;
    for (const auto& [cid, nids] : map_in)
    {
        if ((cid != -1) && this->exclude_bare_cyto && (nids.empty())) continue;
        bool is_valid = this->test_if_valid(image, cytomask_i, cid, "cyto");
        if (!is_valid) continue;
        cytomask_o

    }
}
