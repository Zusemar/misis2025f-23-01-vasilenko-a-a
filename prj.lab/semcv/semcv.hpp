
#pragma once
#ifndef MISIS2025S_3_SEMCV
#define MISIS2025S_3_SEMCV
 
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
 
struct PixelDistributionStats {
    int count{};
    double mean{};
    double variance{};
    double stddev{};
    double minimum{};
    double maximum{};
};
 
std::string strid_from_mat(const cv::Mat& img, const int n = 4);
 
std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst);
 
cv::Mat gen_tgtimg00(const int lev0, const int lev1, const int lev2);
 
cv::Mat add_noise_gau(const cv::Mat& img, const int sigma);

PixelDistributionStats calc_distribution_stats(const cv::Mat& img, const cv::Mat& mask);

cv::Mat draw_histogram_8u(const cv::Mat& img,
                          const cv::Scalar& background_color = cv::Scalar(224, 224, 224),
                          const cv::Scalar& bar_color = cv::Scalar(32, 32, 32));
 
cv::Mat generate_gray_bars_8u_768x30();
 
cv::Mat gamma_correction_8u(const cv::Mat& src_gray_8u, double gamma);
 
#endif  
