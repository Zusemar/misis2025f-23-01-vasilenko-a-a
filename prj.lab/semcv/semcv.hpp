
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

// --- Lab03: Бинаризация ---

// Конвертация в серое одноканальное изображение
cv::Mat to_grayscale(const cv::Mat& img);

// Глобальная бинаризация (простой порог)
cv::Mat global_threshold(const cv::Mat& gray, double threshold, double maxval = 255.0);

// Визуализация маски: наложение полупрозрачной маски на исходное изображение
cv::Mat overlay_mask(const cv::Mat& img, const cv::Mat& mask, 
                     const cv::Scalar& mask_color = cv::Scalar(0, 255, 0), 
                     double alpha = 0.5);

// --- Lab03: Оценка качества бинаризации ---

struct BinaryClassificationMetrics {
    int TP{0};  // True Positive
    int FP{0};  // False Positive
    int FN{0};  // False Negative
    int TN{0};  // True Negative
    
    double TPR() const { return (TP + FN > 0) ? double(TP) / (TP + FN) : 0.0; }  // True Positive Rate (Recall)
    double FPR() const { return (FP + TN > 0) ? double(FP) / (FP + TN) : 0.0; }  // False Positive Rate
    double Precision() const { return (TP + FP > 0) ? double(TP) / (TP + FP) : 0.0; }
    double IoU() const { return (TP + FP + FN > 0) ? double(TP) / (TP + FP + FN) : 0.0; }  // Intersection over Union
    double Accuracy() const { 
        int total = TP + FP + FN + TN;
        return (total > 0) ? double(TP + TN) / total : 0.0;
    }
};

// Вычисление метрик классификации (сравнение предсказанной маски с эталонной)
BinaryClassificationMetrics calc_binary_metrics(const cv::Mat& predicted_mask, 
                                                 const cv::Mat& ground_truth_mask);

#endif  
