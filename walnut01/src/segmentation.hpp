#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace walnut {
struct Parameters {
    std::string filter = "guided";
    std::string method = "regions";
    double intensity_scale = 0.067;
    int radius = 4;
    double epsilon = 0.001;
    double sigma = 1.0;
    double air_factor = 1.0;
    double minimum_material = 0.012;
    double empty_peak = 0.012;
    int core_radius = 3;
    int min_core_area = 4;
    double min_kernel_level = 0.40;
    double max_texture = 0.10;
    double edge_weight = 10.0;
};
struct RegionInfo {
    int area;
    double mean, median, texture;
    int label;
};
struct Result {
    cv::Mat labels;             // uint8: 0 air/background, 1 shell/septa, 2 kernel
    cv::Mat filtered;           // float32, original intensity units
    cv::Mat material;           // binary foreground
    cv::Mat cores;              // eroded foreground used to identify regions
    cv::Mat markers;            // 0 unknown, 1 shell, 2 kernel; before propagation
    double air_threshold = 0;
    double shell_threshold = 0;
    double elapsed_ms = 0;
    bool empty = false;
    std::vector<RegionInfo> regions;
};
Parameters loadParameters(const std::string& file);
void validateParameters(const Parameters& p);
cv::Mat guidedSelfFilter(const cv::Mat& input, int radius, double epsilon);
std::pair<double,double> multiOtsu(const cv::Mat& normalized);
Result segment(const cv::Mat& image, const Parameters& parameters);
} // namespace walnut
