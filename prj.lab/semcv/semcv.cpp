#include <opencv2/opencv.hpp>

#include <semcv/semcv.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <set>
#include <algorithm>

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

// --- Lab04: Сегментация ---

cv::Mat segment_kmeans(const cv::Mat& img, int K, int attempts, cv::TermCriteria criteria) {
    CV_Assert(K > 0 && K <= 255);
    CV_Assert(img.data && !img.empty());
    
    // Подготавливаем данные для K-means
    cv::Mat data;
    if (img.channels() == 1) {
        data = img.reshape(1, img.rows * img.cols);
    } else if (img.channels() == 3) {
        data = img.reshape(1, img.rows * img.cols);
    } else {
        cv::Mat gray = to_grayscale(img);
        data = gray.reshape(1, gray.rows * gray.cols);
    }
    
    data.convertTo(data, CV_32F);
    
    // Выполняем K-means
    cv::Mat labels, centers;
    cv::kmeans(data, K, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, centers);
    
    // Преобразуем метки обратно в изображение
    cv::Mat segmentation = labels.reshape(1, img.rows);
    segmentation.convertTo(segmentation, CV_8UC1);
    
    // Масштабируем значения от 0 до 255 для лучшей визуализации
    cv::Mat scaled;
    segmentation.convertTo(scaled, CV_8UC1, 255.0 / (K - 1), 0);
    
    return segmentation;
}

cv::Mat segment_watershed(const cv::Mat& img, const cv::Mat& markers) {
    CV_Assert(img.data && !img.empty());
    CV_Assert(markers.data && !markers.empty());
    CV_Assert(markers.size() == img.size());
    CV_Assert(markers.type() == CV_32SC1);
    
    cv::Mat gray = to_grayscale(img);
    cv::Mat markers_copy = markers.clone();
    
    // Применяем watershed
    cv::watershed(img, markers_copy);
    
    // Преобразуем результат в 8UC1 (значения -1 и границы становятся 0, остальные - номера регионов)
    cv::Mat segmentation = cv::Mat::zeros(markers.size(), CV_8UC1);
    for (int y = 0; y < markers_copy.rows; ++y) {
        for (int x = 0; x < markers_copy.cols; ++x) {
            int label = markers_copy.at<int>(y, x);
            if (label > 0) {
                segmentation.at<uchar>(y, x) = static_cast<uchar>(label);
            }
        }
    }
    
    return segmentation;
}

cv::Mat visualize_segmentation(const cv::Mat& segmentation_mask, 
                               const cv::Mat& original_img, 
                               double overlay_alpha) {
    CV_Assert(segmentation_mask.data && !segmentation_mask.empty());
    CV_Assert(segmentation_mask.type() == CV_8UC1);
    
    // Генерируем случайные цвета для каждого сегмента
    std::vector<cv::Vec3b> colors(256);
    cv::RNG rng(12345);
    for (int i = 0; i < 256; ++i) {
        colors[i] = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    }
    
    // Раскрашиваем сегментацию
    cv::Mat colored_segmentation(segmentation_mask.size(), CV_8UC3);
    for (int y = 0; y < segmentation_mask.rows; ++y) {
        for (int x = 0; x < segmentation_mask.cols; ++x) {
            uchar label = segmentation_mask.at<uchar>(y, x);
            colored_segmentation.at<cv::Vec3b>(y, x) = colors[label];
        }
    }
    
    // Если есть исходное изображение, накладываем сегментацию с прозрачностью
    if (!original_img.empty() && overlay_alpha > 0.0) {
        cv::Mat img_color;
        if (original_img.channels() == 1) {
            cv::cvtColor(original_img, img_color, cv::COLOR_GRAY2BGR);
        } else {
            img_color = original_img.clone();
        }
        
        cv::Mat result;
        cv::addWeighted(img_color, 1.0 - overlay_alpha, colored_segmentation, overlay_alpha, 0.0, result);
        return result;
    }
    
    return colored_segmentation;
}

cv::Mat visualize_segmentation_errors(const cv::Mat& predicted, 
                                      const cv::Mat& ground_truth,
                                      const cv::Mat& original_img,
                                      double overlay_alpha) {
    CV_Assert(predicted.size() == ground_truth.size());
    CV_Assert(predicted.type() == CV_8UC1);
    CV_Assert(ground_truth.type() == CV_8UC1);
    
    cv::Mat error_vis(predicted.size(), CV_8UC3);
    
    for (int y = 0; y < predicted.rows; ++y) {
        for (int x = 0; x < predicted.cols; ++x) {
            uchar pred = predicted.at<uchar>(y, x);
            uchar gt = ground_truth.at<uchar>(y, x);
            
            if (pred == gt) {
                // Правильно сегментированные пиксели - зеленые
                error_vis.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
            } else if (pred > 0 && gt == 0) {
                // Ложные срабатывания (FP) - красные
                error_vis.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
            } else if (pred == 0 && gt > 0) {
                // Пропущенные пиксели (FN) - синие
                error_vis.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
            } else {
                // Неправильная классификация (разные классы) - желтые
                error_vis.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 255);
            }
        }
    }
    
    // Если есть исходное изображение, накладываем с прозрачностью
    if (!original_img.empty() && overlay_alpha > 0.0) {
        cv::Mat img_color;
        if (original_img.channels() == 1) {
            cv::cvtColor(original_img, img_color, cv::COLOR_GRAY2BGR);
        } else {
            img_color = original_img.clone();
        }
        
        cv::Mat result;
        cv::addWeighted(img_color, 1.0 - overlay_alpha, error_vis, overlay_alpha, 0.0, result);
        return result;
    }
    
    return error_vis;
}

// --- Lab04: Оценка качества сегментации ---

void SegmentationMetrics::compute() {
    if (num_classes == 0) return;
    
    Precision.resize(num_classes);
    Recall.resize(num_classes);
    IoU.resize(num_classes);
    F1.resize(num_classes);
    
    double sum_iou = 0.0;
    double sum_f1 = 0.0;
    int total_pixels = 0;
    int correct_pixels = 0;
    
    for (int i = 0; i < num_classes; ++i) {
        int tp = TP[i];
        int fp = FP[i];
        int fn = FN[i];
        
        Precision[i] = (tp + fp > 0) ? double(tp) / (tp + fp) : 0.0;
        Recall[i] = (tp + fn > 0) ? double(tp) / (tp + fn) : 0.0;
        IoU[i] = (tp + fp + fn > 0) ? double(tp) / (tp + fp + fn) : 0.0;
        F1[i] = (Precision[i] + Recall[i] > 0) ? 2.0 * Precision[i] * Recall[i] / (Precision[i] + Recall[i]) : 0.0;
        
        sum_iou += IoU[i];
        sum_f1 += F1[i];
        total_pixels += tp + fp + fn;
        correct_pixels += tp;
    }
    
    MeanIoU = (num_classes > 0) ? sum_iou / num_classes : 0.0;
    MeanF1 = (num_classes > 0) ? sum_f1 / num_classes : 0.0;
    OverallAccuracy = (total_pixels > 0) ? double(correct_pixels) / total_pixels : 0.0;
}

SegmentationMetrics calc_segmentation_metrics(const cv::Mat& predicted_segmentation,
                                              const cv::Mat& ground_truth_segmentation) {
    CV_Assert(predicted_segmentation.size() == ground_truth_segmentation.size());
    CV_Assert(predicted_segmentation.type() == CV_8UC1);
    CV_Assert(ground_truth_segmentation.type() == CV_8UC1);
    
    SegmentationMetrics metrics;
    
    // Находим все уникальные классы в обеих сегментациях
    std::set<int> all_classes;
    for (int y = 0; y < ground_truth_segmentation.rows; ++y) {
        for (int x = 0; x < ground_truth_segmentation.cols; ++x) {
            int gt_class = ground_truth_segmentation.at<uchar>(y, x);
            int pred_class = predicted_segmentation.at<uchar>(y, x);
            if (gt_class > 0) all_classes.insert(gt_class);
            if (pred_class > 0) all_classes.insert(pred_class);
        }
    }
    
    metrics.num_classes = static_cast<int>(all_classes.size());
    metrics.class_ids.assign(all_classes.begin(), all_classes.end());
    metrics.TP.resize(metrics.num_classes, 0);
    metrics.FP.resize(metrics.num_classes, 0);
    metrics.FN.resize(metrics.num_classes, 0);
    
    // Вычисляем TP, FP, FN для каждого класса
    for (int y = 0; y < predicted_segmentation.rows; ++y) {
        for (int x = 0; x < predicted_segmentation.cols; ++x) {
            int pred = predicted_segmentation.at<uchar>(y, x);
            int gt = ground_truth_segmentation.at<uchar>(y, x);
            
            // Для каждого класса проверяем, правильно ли он классифицирован
            for (int i = 0; i < metrics.num_classes; ++i) {
                int class_id = metrics.class_ids[i];
                
                bool pred_is_class = (pred == class_id);
                bool gt_is_class = (gt == class_id);
                
                if (pred_is_class && gt_is_class) {
                    metrics.TP[i]++;
                } else if (pred_is_class && !gt_is_class) {
                    metrics.FP[i]++;
                } else if (!pred_is_class && gt_is_class) {
                    metrics.FN[i]++;
                }
            }
        }
    }
    
    metrics.compute();
    return metrics;
}

// --- Lab05: Детектирование объектов ---

double calculate_iou(const cv::Rect& box1, const cv::Rect& box2) {
    // Вычисляем пересечение
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) {
        return 0.0;  // Нет пересечения
    }
    
    int intersection_area = (x2 - x1) * (y2 - y1);
    int area1 = box1.width * box1.height;
    int area2 = box2.width * box2.height;
    int union_area = area1 + area2 - intersection_area;
    
    return (union_area > 0) ? double(intersection_area) / union_area : 0.0;
}

double estimate_detection_confidence(const cv::Mat& img,
                                    const cv::Rect& bbox,
                                    const cv::Mat& binary_mask) {
    if (bbox.width <= 0 || bbox.height <= 0) return 0.0;
    
    // Проверяем размер маски
    if (binary_mask.size() != img.size()) {
        // Если маска не соответствует размеру изображения, создаем простую оценку
        return 0.5;  // Средняя достоверность
    }
    
    // Обрезаем bbox до границ изображения
    cv::Rect valid_bbox = bbox & cv::Rect(0, 0, img.cols, img.rows);
    if (valid_bbox.width <= 0 || valid_bbox.height <= 0) return 0.0;
    
    cv::Mat gray = to_grayscale(img);
    cv::Mat roi = gray(valid_bbox);
    cv::Mat mask_roi = binary_mask(valid_bbox);
    
    // 1. Оценка контраста (разница между объектом и фоном)
    cv::Scalar mean_obj, mean_bg, stddev_obj, stddev_bg;
    cv::meanStdDev(roi, mean_obj, stddev_obj, mask_roi);
    cv::meanStdDev(roi, mean_bg, stddev_bg, ~mask_roi);
    
    double contrast = std::abs(mean_obj[0] - mean_bg[0]) / 255.0;
    contrast = std::min(contrast, 1.0);
    
    // 2. Оценка размера (нормализованная площадь)
    double area_ratio = double(cv::countNonZero(mask_roi)) / (roi.rows * roi.cols);
    
    // 3. Оценка компактности (отношение площади к периметру)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double compactness = 0.0;
    if (!contours.empty()) {
        double area = cv::contourArea(contours[0]);
        double perimeter = cv::arcLength(contours[0], true);
        if (perimeter > 0) {
            compactness = 4.0 * M_PI * area / (perimeter * perimeter);
        }
    }
    
    // Комбинированная оценка достоверности
    double confidence = 0.4 * contrast + 0.3 * area_ratio + 0.3 * compactness;
    return std::min(std::max(confidence, 0.0), 1.0);
}

// Вспомогательная функция для вычисления уверенности на основе fillRatio и compactness
static double calculate_confidence_from_shape(const cv::Mat& mask, const cv::Rect& bbox) {
    int maskArea = cv::countNonZero(mask);
    int bboxArea = bbox.area();
    
    if (bboxArea == 0) return 0.0;
    
    double fillRatio = static_cast<double>(maskArea) / bboxArea;
    
    // Вычисляем компактность
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    double compactness = 0.0;
    if (!contours.empty()) {
        double area = cv::contourArea(contours[0]);
        double perimeter = cv::arcLength(contours[0], true);
        if (perimeter > 0) {
            compactness = (4.0 * CV_PI * area) / (perimeter * perimeter);
        }
    }
    
    // Комбинируем метрики
    double confidence = 0.6 * fillRatio + 0.4 * compactness;
    return std::min(std::max(confidence, 0.0), 1.0);
}

// Объединение дублирующихся детекций
static std::vector<DetectedObject> merge_duplicate_detections(std::vector<DetectedObject>& detections) {
    std::vector<DetectedObject> merged;
    std::vector<bool> mergedFlag(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (mergedFlag[i]) continue;
        
        DetectedObject current = detections[i];
        std::vector<int> toMerge = {static_cast<int>(i)};
        
        // Ищем перекрывающиеся детекции
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (mergedFlag[j]) continue;
            
            cv::Rect intersection = current.bbox & detections[j].bbox;
            double unionArea = current.bbox.area() + detections[j].bbox.area() - intersection.area();
            double overlap = unionArea > 0 ? intersection.area() / unionArea : 0.0;
            
            if (overlap > 0.5) { // Порог перекрытия
                toMerge.push_back(static_cast<int>(j));
            }
        }
        
        // Объединяем детекции
        if (toMerge.size() > 1) {
            // Находим общий bounding box
            cv::Rect mergedBox = detections[toMerge[0]].bbox;
            double totalConfidence = 0;
            
            for (size_t k = 1; k < toMerge.size(); k++) {
                mergedBox |= detections[toMerge[k]].bbox;
                totalConfidence += detections[toMerge[k]].confidence;
            }
            
            double avgConfidence = totalConfidence / toMerge.size();
            
            // Создаем объединенную маску
            cv::Mat mergedMask = cv::Mat::zeros(detections[toMerge[0]].mask.size(), CV_8UC1);
            for (int idx : toMerge) {
                if (!detections[idx].mask.empty()) {
                    mergedMask = mergedMask | detections[idx].mask;
                }
            }
            
            current = DetectedObject(mergedBox, avgConfidence, current.scaleLevel, mergedMask);
        }
        
        merged.push_back(current);
        for (int idx : toMerge) {
            mergedFlag[idx] = true;
        }
    }
    
    return merged;
}

std::vector<DetectedObject> detect_objects(const cv::Mat& img,
                                           int min_area,
                                           int max_area,
                                           const std::vector<double>& scale_factors) {
    std::vector<DetectedObject> all_detections;
    
    cv::Mat gray = to_grayscale(img);
    
    // Многомасштабный анализ
    for (double scale : scale_factors) {
        cv::Mat scaled_img;
        if (std::abs(scale - 1.0) < 1e-6) {
            scaled_img = gray.clone();
        } else {
            cv::resize(gray, scaled_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
        }
        
        // Улучшаем контраст
        cv::Mat enhanced;
        cv::equalizeHist(scaled_img, enhanced);
        
        // Применяем адаптивный порог
        cv::Mat binary;
        cv::adaptiveThreshold(enhanced, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY_INV, 11, 2);
        
        // Морфологические операции для улучшения
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
        
        // Находим контуры
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        
        int scaled_min_area = static_cast<int>(min_area * scale * scale);
        int scaled_max_area = static_cast<int>(max_area * scale * scale);
        double connectivityThreshold = 0.5;  // Увеличен порог для более строгой фильтрации
        
        // Анализируем каждый контур
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area < scaled_min_area || area > scaled_max_area) continue;
            
            // Вычисляем ограничивающий прямоугольник
            cv::Rect bbox_scaled = cv::boundingRect(contours[i]);
            
            // Создаем маску для компоненты
            cv::Mat mask = cv::Mat::zeros(binary.size(), CV_8UC1);
            cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
            
            // Вычисляем уверенность детекции
            double confidence = calculate_confidence_from_shape(mask, bbox_scaled);
            
            if (confidence > connectivityThreshold) {
                // Масштабируем обратно к исходному размеру
                cv::Rect bbox(static_cast<int>(bbox_scaled.x / scale),
                             static_cast<int>(bbox_scaled.y / scale),
                             static_cast<int>(bbox_scaled.width / scale),
                             static_cast<int>(bbox_scaled.height / scale));
                
                bbox &= cv::Rect(0, 0, img.cols, img.rows);
                
                if (bbox.width >= 5 && bbox.height >= 5) {
                    // Масштабируем маску обратно
                    cv::Mat mask_scaled;
                    if (std::abs(scale - 1.0) > 1e-6) {
                        cv::resize(mask, mask_scaled, img.size(), 0, 0, cv::INTER_NEAREST);
                    } else {
                        mask_scaled = mask;
                    }
                    
                    int scaleLevel = static_cast<int>(scale * 10);
                    all_detections.emplace_back(bbox, confidence, scaleLevel, mask_scaled);
                }
            }
        }
    }
    
    // Устраняем дубликаты
    return merge_duplicate_detections(all_detections);
}

cv::Mat visualize_detections(const cv::Mat& img,
                            const std::vector<DetectedObject>& detections,
                            const cv::Scalar& color,
                            double thickness) {
    cv::Mat result = img.clone();
    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    
    for (const auto& det : detections) {
        // Цвет зависит от достоверности: от красного (низкая) к зеленому (высокая)
        int red = cvRound(255 * (1.0 - det.confidence));
        int green = cvRound(255 * det.confidence);
        cv::Scalar det_color(0, green, red);  // BGR формат
        
        cv::rectangle(result, det.bbox, det_color, static_cast<int>(thickness));
        
        // Добавляем текст с confidence
        char conf_text[32];
        snprintf(conf_text, sizeof(conf_text), "%.2f", det.confidence);
        cv::putText(result, conf_text, cv::Point(det.bbox.x, det.bbox.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Рисуем центр, если он определен
        if (det.center.x > 0 && det.center.y > 0) {
            cv::circle(result, det.center, 3, det_color, cv::FILLED);
        }
    }
    
    return result;
}

cv::Mat visualize_detection_errors(const cv::Mat& img,
                                 const std::vector<DetectedObject>& predicted,
                                 const std::vector<DetectedObject>& ground_truth,
                                 double iou_threshold) {
    cv::Mat result = img.clone();
    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    
    // Отмечаем, какие ground truth объекты были найдены
    std::vector<bool> gt_matched(ground_truth.size(), false);
    
    // Рисуем правильные детекции (TP) - зеленые
    for (const auto& pred : predicted) {
        bool matched = false;
        for (size_t i = 0; i < ground_truth.size(); ++i) {
            if (!gt_matched[i]) {
                double iou = calculate_iou(pred.bbox, ground_truth[i].bbox);
                if (iou >= iou_threshold) {
                    cv::rectangle(result, pred.bbox, cv::Scalar(0, 255, 0), 2);  // Зеленый - TP
                    gt_matched[i] = true;
                    matched = true;
                    break;
                }
            }
        }
        
        // Ложные срабатывания (FP) - красные
        if (!matched) {
            cv::rectangle(result, pred.bbox, cv::Scalar(0, 0, 255), 2);  // Красный - FP
        }
    }
    
    // Пропущенные объекты (FN) - синие
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        if (!gt_matched[i]) {
            cv::rectangle(result, ground_truth[i].bbox, cv::Scalar(255, 0, 0), 2);  // Синий - FN
        }
    }
    
    return result;
}

DetectionMetrics calc_detection_metrics(const std::vector<DetectedObject>& predicted,
                                       const std::vector<DetectedObject>& ground_truth,
                                       double iou_threshold) {
    DetectionMetrics metrics;
    
    if (predicted.empty() && ground_truth.empty()) {
        return metrics;  // Оба пустые - идеальный результат
    }
    
    std::vector<bool> pred_matched(predicted.size(), false);
    std::vector<bool> gt_matched(ground_truth.size(), false);
    std::vector<double> ious;
    
    // Находим совпадения
    for (size_t i = 0; i < predicted.size(); ++i) {
        double best_iou = 0.0;
        size_t best_gt_idx = ground_truth.size();
        
        for (size_t j = 0; j < ground_truth.size(); ++j) {
            if (!gt_matched[j]) {
                double iou = calculate_iou(predicted[i].bbox, ground_truth[j].bbox);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_gt_idx = j;
                }
            }
        }
        
        if (best_iou >= iou_threshold) {
            metrics.TP++;
            pred_matched[i] = true;
            gt_matched[best_gt_idx] = true;
            ious.push_back(best_iou);
        } else {
            metrics.FP++;
        }
    }
    
    // Подсчитываем пропущенные объекты (FN)
    for (size_t j = 0; j < ground_truth.size(); ++j) {
        if (!gt_matched[j]) {
            metrics.FN++;
        }
    }
    
    // Вычисляем средний IoU
    if (!ious.empty()) {
        double sum_iou = 0.0;
        for (double iou : ious) {
            sum_iou += iou;
        }
        metrics.MeanIoU = sum_iou / ious.size();
        metrics.detection_ious = ious;
    }
    
    return metrics;
}

// --- Lab06: Векторизация границ объектов ---

std::vector<cv::Point> simplify_contour(const std::vector<cv::Point>& contour, double epsilon) {
    if (contour.size() < 3) return contour;
    
    std::vector<cv::Point> simplified;
    cv::approxPolyDP(contour, simplified, epsilon, true);
    return simplified;
}

std::vector<VectorizedContour> vectorize_object_boundaries(const cv::Mat& binary_mask,
                                                           double min_area,
                                                           double epsilon) {
    std::vector<VectorizedContour> result;
    
    if (binary_mask.empty() || binary_mask.channels() != 1) {
        return result;
    }
    
    // Новый подход: улучшенная обработка бинарной маски
    cv::Mat binary_processed = binary_mask.clone();
    
    // Применяем морфологические операции для улучшения границ
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary_processed, binary_processed, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    cv::morphologyEx(binary_processed, binary_processed, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    
    // Находим контуры с использованием RETR_TREE для получения иерархии
    // Это позволяет различать внешние и внутренние контуры
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_processed, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    
    // Фильтруем и обрабатываем контуры
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contours[i].size() < 3) continue;
        
        // Пропускаем внутренние контуры (дочерние)
        if (hierarchy[i][3] >= 0) continue;  // Если есть родитель, это внутренний контур
        
        double area = cv::contourArea(contours[i]);
        if (area < min_area) continue;
        
        // Вычисляем периметр для определения подходящего epsilon
        double perimeter = cv::arcLength(contours[i], true);
        double adaptive_epsilon = std::max(epsilon, perimeter * 0.01);  // Адаптивный epsilon
        
        // Упрощаем контур с адаптивным epsilon
        std::vector<cv::Point> simplified = simplify_contour(contours[i], adaptive_epsilon);
        
        // Проверяем, что упрощенный контур валиден
        if (simplified.size() < 3) continue;
        
        // Создаем векторный контур
        VectorizedContour vc(simplified);
        
        // Дополнительная проверка: контур должен быть достаточно компактным
        double compactness = 4.0 * M_PI * vc.area / (vc.perimeter * vc.perimeter);
        if (compactness > 0.01) {  // Фильтруем слишком вытянутые контуры
            result.push_back(vc);
        }
    }
    
    return result;
}

bool save_vectorized_contours(const std::string& filepath,
                              const std::vector<VectorizedContour>& contours) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    for (const auto& contour : contours) {
        for (size_t i = 0; i < contour.points.size(); ++i) {
            file << contour.points[i].x << "," << contour.points[i].y;
            if (i < contour.points.size() - 1) {
                file << " ";
            }
        }
        file << "\n";
    }
    
    file.close();
    return true;
}

std::vector<VectorizedContour> load_vectorized_contours(const std::string& filepath) {
    std::vector<VectorizedContour> result;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return result;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<cv::Point> points;
        std::istringstream iss(line);
        std::string token;
        
        while (std::getline(iss, token, ' ')) {
            if (token.empty()) continue;
            
            size_t comma_pos = token.find(',');
            if (comma_pos != std::string::npos) {
                int x = std::stoi(token.substr(0, comma_pos));
                int y = std::stoi(token.substr(comma_pos + 1));
                points.push_back(cv::Point(x, y));
            }
        }
        
        if (points.size() >= 3) {
            result.emplace_back(points);
        }
    }
    
    file.close();
    return result;
}

cv::Mat visualize_vectorized_contours(const cv::Mat& img,
                                     const std::vector<VectorizedContour>& contours,
                                     const cv::Scalar& color,
                                     int thickness,
                                     bool fill) {
    cv::Mat result = img.clone();
    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    
    for (const auto& contour : contours) {
        if (contour.points.size() < 3) continue;
        
        std::vector<std::vector<cv::Point>> draw_contours = {contour.points};
        
        if (fill) {
            cv::fillPoly(result, draw_contours, color);
        } else {
            cv::drawContours(result, draw_contours, -1, color, thickness);
        }
    }
    
    return result;
}

double hausdorff_distance(const std::vector<cv::Point>& contour1,
                         const std::vector<cv::Point>& contour2) {
    if (contour1.empty() || contour2.empty()) {
        return std::numeric_limits<double>::max();
    }
    
    double max_dist = 0.0;
    
    // Для каждой точки в contour1 находим минимальное расстояние до contour2
    for (const auto& p1 : contour1) {
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& p2 : contour2) {
            double dist = cv::norm(p1 - p2);
            min_dist = std::min(min_dist, dist);
        }
        max_dist = std::max(max_dist, min_dist);
    }
    
    // Для каждой точки в contour2 находим минимальное расстояние до contour1
    for (const auto& p2 : contour2) {
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& p1 : contour1) {
            double dist = cv::norm(p1 - p2);
            min_dist = std::min(min_dist, dist);
        }
        max_dist = std::max(max_dist, min_dist);
    }
    
    return max_dist;
}

double average_contour_distance(const std::vector<cv::Point>& contour1,
                                const std::vector<cv::Point>& contour2) {
    if (contour1.empty() || contour2.empty()) {
        return std::numeric_limits<double>::max();
    }
    
    double total_dist = 0.0;
    int count = 0;
    
    // Для каждой точки в contour1 находим минимальное расстояние до contour2
    for (const auto& p1 : contour1) {
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& p2 : contour2) {
            double dist = cv::norm(p1 - p2);
            min_dist = std::min(min_dist, dist);
        }
        total_dist += min_dist;
        count++;
    }
    
    // Для каждой точки в contour2 находим минимальное расстояние до contour1
    for (const auto& p2 : contour2) {
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& p1 : contour1) {
            double dist = cv::norm(p1 - p2);
            min_dist = std::min(min_dist, dist);
        }
        total_dist += min_dist;
        count++;
    }
    
    return (count > 0) ? total_dist / count : std::numeric_limits<double>::max();
}

double contour_iou(const std::vector<cv::Point>& contour1,
                  const std::vector<cv::Point>& contour2,
                  const cv::Size& image_size) {
    if (contour1.empty() || contour2.empty()) {
        return 0.0;
    }
    
    // Создаем маски для контуров
    cv::Mat mask1 = cv::Mat::zeros(image_size, CV_8UC1);
    cv::Mat mask2 = cv::Mat::zeros(image_size, CV_8UC1);
    
    std::vector<std::vector<cv::Point>> contours1 = {contour1};
    std::vector<std::vector<cv::Point>> contours2 = {contour2};
    
    cv::fillPoly(mask1, contours1, cv::Scalar(255));
    cv::fillPoly(mask2, contours2, cv::Scalar(255));
    
    // Вычисляем IoU
    cv::Mat intersection, union_mask;
    cv::bitwise_and(mask1, mask2, intersection);
    cv::bitwise_or(mask1, mask2, union_mask);
    
    double intersection_area = cv::countNonZero(intersection);
    double union_area = cv::countNonZero(union_mask);
    
    return (union_area > 0) ? intersection_area / union_area : 0.0;
}

VectorizationMetrics calc_vectorization_metrics(const std::vector<VectorizedContour>& predicted,
                                                const std::vector<VectorizedContour>& ground_truth,
                                                double distance_threshold) {
    VectorizationMetrics metrics;
    
    if (predicted.empty() && ground_truth.empty()) {
        return metrics;
    }
    
    // Определяем размер изображения (берем максимальные координаты)
    int max_x = 0, max_y = 0;
    for (const auto& c : predicted) {
        for (const auto& p : c.points) {
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
        }
    }
    for (const auto& c : ground_truth) {
        for (const auto& p : c.points) {
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
        }
    }
    cv::Size img_size(max_x + 1, max_y + 1);
    
    // Отмечаем, какие ground truth контуры были найдены
    std::vector<bool> gt_matched(ground_truth.size(), false);
    std::vector<double> hausdorff_dists;
    std::vector<double> avg_dists;
    std::vector<double> ious;
    
    // Ищем совпадения для каждого предсказанного контура
    for (const auto& pred : predicted) {
        bool matched = false;
        double best_hausdorff = std::numeric_limits<double>::max();
        double best_avg_dist = std::numeric_limits<double>::max();
        double best_iou = 0.0;
        size_t best_gt_idx = 0;
        
        for (size_t i = 0; i < ground_truth.size(); ++i) {
            if (gt_matched[i]) continue;
            
            double hd = hausdorff_distance(pred.points, ground_truth[i].points);
            double avg_d = average_contour_distance(pred.points, ground_truth[i].points);
            double iou = contour_iou(pred.points, ground_truth[i].points, img_size);
            
            // Используем комбинацию метрик для определения совпадения
            if (avg_d <= distance_threshold && iou > 0.3) {
                if (avg_d < best_avg_dist) {
                    best_hausdorff = hd;
                    best_avg_dist = avg_d;
                    best_iou = iou;
                    best_gt_idx = i;
                    matched = true;
                }
            }
        }
        
        if (matched) {
            metrics.TP++;
            gt_matched[best_gt_idx] = true;
            hausdorff_dists.push_back(best_hausdorff);
            avg_dists.push_back(best_avg_dist);
            ious.push_back(best_iou);
        } else {
            metrics.FP++;
        }
    }
    
    // Подсчитываем пропущенные контуры (FN)
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        if (!gt_matched[i]) {
            metrics.FN++;
        }
    }
    
    // Вычисляем средние метрики
    if (!hausdorff_dists.empty()) {
        double sum_hd = 0.0;
        for (double hd : hausdorff_dists) {
            sum_hd += hd;
        }
        metrics.MeanHausdorffDistance = sum_hd / hausdorff_dists.size();
    }
    
    if (!avg_dists.empty()) {
        double sum_avg = 0.0;
        for (double avg : avg_dists) {
            sum_avg += avg;
        }
        metrics.MeanContourDistance = sum_avg / avg_dists.size();
    }
    
    if (!ious.empty()) {
        double sum_iou = 0.0;
        for (double iou : ious) {
            sum_iou += iou;
        }
        metrics.MeanIoU = sum_iou / ious.size();
    }
    
    return metrics;
}

cv::Mat visualize_vectorization_errors(const cv::Mat& img,
                                     const std::vector<VectorizedContour>& predicted,
                                     const std::vector<VectorizedContour>& ground_truth,
                                     double distance_threshold) {
    cv::Mat result = img.clone();
    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    
    // Определяем размер изображения
    int max_x = result.cols, max_y = result.rows;
    cv::Size img_size(max_x, max_y);
    
    // Отмечаем, какие ground truth контуры были найдены
    std::vector<bool> gt_matched(ground_truth.size(), false);
    
    // Рисуем правильные контуры (TP) - зеленые
    for (const auto& pred : predicted) {
        bool matched = false;
        for (size_t i = 0; i < ground_truth.size(); ++i) {
            if (!gt_matched[i]) {
                double avg_d = average_contour_distance(pred.points, ground_truth[i].points);
                double iou = contour_iou(pred.points, ground_truth[i].points, img_size);
                
                if (avg_d <= distance_threshold && iou > 0.3) {
                    std::vector<std::vector<cv::Point>> draw_contours = {pred.points};
                    cv::drawContours(result, draw_contours, -1, cv::Scalar(0, 255, 0), 2);  // Зеленый - TP
                    gt_matched[i] = true;
                    matched = true;
                    break;
                }
            }
        }
        
        // Рисуем ложные контуры (FP) - красные
        if (!matched) {
            std::vector<std::vector<cv::Point>> draw_contours = {pred.points};
            cv::drawContours(result, draw_contours, -1, cv::Scalar(0, 0, 255), 2);  // Красный - FP
        }
    }
    
    // Рисуем пропущенные контуры (FN) - синие
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        if (!gt_matched[i]) {
            std::vector<std::vector<cv::Point>> draw_contours = {ground_truth[i].points};
            cv::drawContours(result, draw_contours, -1, cv::Scalar(255, 0, 0), 2);  // Синий - FN
        }
    }
    
    return result;
}
