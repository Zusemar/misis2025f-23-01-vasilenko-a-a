#include <opencv2/opencv.hpp>

#include <semcv/semcv.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cmath>

using namespace std::string_literals;

// --- как у вас ---
std::string strid_from_mat(const cv::Mat& img, const int n) {
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    int depth = img.depth();

    std::string type_name;
    switch (depth) {
        case CV_8U:  type_name = "uint08"; break;
        case CV_8S:  type_name = "sint08"; break;
        case CV_16U: type_name = "uint16"; break;
        case CV_16S: type_name = "sint16"; break;
        case CV_32S: type_name = "sint32"; break;
        case CV_32F: type_name = "real32"; break;
        case CV_64F: type_name = "real64"; break;
        default:     type_name = "unknown"; break;
    }

    std::ostringstream result;
    result << std::setw(n) << std::setfill('0') << width << "x"
           << std::setw(n) << std::setfill('0') << height << "."
           << channels << "." << type_name;
    return result.str();
}

std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst) {
    std::vector<std::filesystem::path> file_paths;
    std::ifstream file(path_lst);
    if (!file.is_open()) return file_paths;

    std::string filename;
    std::filesystem::path base_dir = path_lst.parent_path();
    while (std::getline(file, filename)) {
        if (filename.empty()) continue;
        // trim
        auto l = filename.find_first_not_of(" \t\r\n");
        auto r = filename.find_last_not_of(" \t\r\n");
        if (l == std::string::npos) continue;
        filename = filename.substr(l, r - l + 1);
        if (filename.empty()) continue;

        file_paths.push_back(base_dir / filename);
    }
    return file_paths;
}

// --- задание 1: генерация тестового изображения ---
cv::Mat gen_tgtimg00(const int lev0, const int lev1, const int lev2) {
    CV_Assert(0 <= lev0 && lev0 <= 255);
    CV_Assert(0 <= lev1 && lev1 <= 255);
    CV_Assert(0 <= lev2 && lev2 <= 255);

    cv::Mat image(256, 256, CV_8UC1, cv::Scalar(lev0));

    const int square_size = 209;
    const int square_x = (256 - square_size) / 2; // 23
    const int square_y = (256 - square_size) / 2; // 23
    image(cv::Rect(square_x, square_y, square_size, square_size)).setTo(uchar(lev1));

    const cv::Point c(128, 128);
    const int r = 83;
    cv::circle(image, c, r, cv::Scalar(lev2), cv::FILLED);

    return image;
}

// --- задание 2: аддитивный нормальный шум (несмещённый, std = sigma) ---
cv::Mat add_noise_gau(const cv::Mat& img, const int sigma) {
    CV_Assert(img.data && img.type() == CV_8UC1);
    CV_Assert(sigma >= 0);

    if (sigma == 0) return img.clone();

    cv::Mat img32f, noise32f(img.size(), CV_32F);
    img.convertTo(img32f, CV_32F);
    cv::randn(noise32f, 0.0, static_cast<double>(sigma)); // N(0, sigma^2)

    cv::Mat noisy32f = img32f + noise32f;
    cv::Mat noisy8u;
    noisy32f.convertTo(noisy8u, CV_8U); // с насыщением к [0..255]
    return noisy8u;
}

// --- гистограмма 8u: квадрат 256x256, максимум столбца 250, цвета настраиваемые ---
cv::Mat draw_histogram_8u(const cv::Mat& img,
                          const cv::Scalar& background_color,
                          const cv::Scalar& bar_color) {
    CV_Assert(img.data && img.type() == CV_8UC1);

    // 1) считаем гистограмму
    int histSize = 256;
    float range[] = {0.f, 256.f};
    const float* ranges[] = { range };
    cv::Mat hist;
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, ranges, true, false);

    // 2) нормируем высоты в [0..250]
    double maxVal = 0;
    cv::minMaxLoc(hist, nullptr, &maxVal);
    cv::Mat histN;
    if (maxVal > 0) {
        histN = hist * (250.0 / maxVal);
    } else {
        histN = cv::Mat::zeros(hist.size(), hist.type());
    }

    // 3) рисуем
    cv::Mat canvas(256, 256, CV_8UC3);
    canvas.setTo(background_color);

    for (int x = 0; x < 256; ++x) {
        int h = cvRound(histN.at<float>(x));
        if (h <= 0) continue;
        // столбец высотой h, от низа (y=255) вверх
        cv::rectangle(canvas,
                      cv::Point(x, 255),
                      cv::Point(x, 255 - std::min(h, 250)),
                      bar_color,
                      cv::FILLED);
    }
    return canvas;
}

// --- утилитарные функции (как у вас) ---
cv::Mat generate_gray_bars_8u_768x30() {
    const int width = 768, height = 30, barWidth = 3;
    cv::Mat img(height, width, CV_8UC1);
    for (int x = 0; x < width; ++x) {
        int barIndex = x / barWidth; // 0..255
        barIndex = std::clamp(barIndex, 0, 255);
        uchar v = static_cast<uchar>(barIndex);
        for (int y = 0; y < height; ++y) img.at<uchar>(y, x) = v;
    }
    return img;
}

cv::Mat gamma_correction_8u(const cv::Mat& src, double gamma) {
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(gamma > 0.0);

    cv::Mat lut(1, 256, CV_8U);
    uchar* p = lut.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, 1.0 / gamma) * 255.0);
    }
    cv::Mat dst;
    cv::LUT(src, lut, dst);
    return dst;
}

// --- задание 2.1: статистики по маске ---
PixelDistributionStats calc_distribution_stats(const cv::Mat& img, const cv::Mat& mask) {
    CV_Assert(img.data && img.type() == CV_8UC1);
    CV_Assert(mask.empty() || (mask.size() == img.size() && mask.type() == CV_8UC1));

    PixelDistributionStats s{};
    double sum = 0.0, sum2 = 0.0;
    double mn = std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    int count = 0;

    const int rows = img.rows, cols = img.cols;
    for (int y = 0; y < rows; ++y) {
        const uchar* ip = img.ptr<uchar>(y);
        const uchar* mp = mask.empty() ? nullptr : mask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            if (mp && mp[x] == 0) continue; // берём только ненулевые в маске
            double v = ip[x];
            sum += v;
            sum2 += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            ++count;
        }
    }

    if (count == 0) {
        s.count = 0;
        s.mean = s.variance = s.stddev = s.minimum = s.maximum = std::numeric_limits<double>::quiet_NaN();
        return s;
    }

    double mean = sum / count;
    double var = std::max(0.0, sum2 / count - mean * mean); // дисперсия (несмещённая можно по желанию)
    double stddev = std::sqrt(var);

    s.count = count;
    s.mean = mean;
    s.variance = var;
    s.stddev = stddev;
    s.minimum = mn;
    s.maximum = mx;
    return s;
}

// --- Lab03: Бинаризация ---

cv::Mat to_grayscale(const cv::Mat& img) {
    if (img.channels() == 1) {
        return img.clone();
    }
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        // fallback: берём первый канал
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        gray = channels[0].clone();
    }
    return gray;
}

cv::Mat global_threshold(const cv::Mat& gray, double threshold, double maxval) {
    CV_Assert(gray.type() == CV_8UC1);
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, maxval, cv::THRESH_BINARY);
    return binary;
}

cv::Mat overlay_mask(const cv::Mat& img, const cv::Mat& mask, 
                     const cv::Scalar& mask_color, double alpha) {
    CV_Assert(img.size() == mask.size());
    CV_Assert(mask.type() == CV_8UC1);
    
    // Конвертируем исходное изображение в цветное (BGR), если нужно
    cv::Mat img_color;
    if (img.channels() == 1) {
        cv::cvtColor(img, img_color, cv::COLOR_GRAY2BGR);
    } else {
        img_color = img.clone();
    }
    
    // Создаём цветную маску
    cv::Mat colored_mask;
    cv::cvtColor(mask, colored_mask, cv::COLOR_GRAY2BGR);
    colored_mask.setTo(mask_color, mask > 0);
    
    // Накладываем маску с прозрачностью
    cv::Mat result;
    cv::addWeighted(img_color, 1.0 - alpha, colored_mask, alpha, 0.0, result);
    
    return result;
}

// --- Lab03: Оценка качества бинаризации ---

BinaryClassificationMetrics calc_binary_metrics(const cv::Mat& predicted_mask, 
                                                 const cv::Mat& ground_truth_mask) {
    CV_Assert(predicted_mask.size() == ground_truth_mask.size());
    CV_Assert(predicted_mask.type() == CV_8UC1);
    CV_Assert(ground_truth_mask.type() == CV_8UC1);
    
    BinaryClassificationMetrics m{};
    
    const int rows = predicted_mask.rows;
    const int cols = predicted_mask.cols;
    
    for (int y = 0; y < rows; ++y) {
        const uchar* pred = predicted_mask.ptr<uchar>(y);
        const uchar* gt = ground_truth_mask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            bool pred_pos = (pred[x] > 0);
            bool gt_pos = (gt[x] > 0);
            
            if (pred_pos && gt_pos) {
                m.TP++;
            } else if (pred_pos && !gt_pos) {
                m.FP++;
            } else if (!pred_pos && gt_pos) {
                m.FN++;
            } else {
                m.TN++;
            }
        }
    }
    
    return m;
}
