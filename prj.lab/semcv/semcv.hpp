
#pragma once
#ifndef MISIS2025S_3_SEMCV
#define MISIS2025S_3_SEMCV
 
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
 
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

// --- Lab04: Сегментация ---

// K-means сегментация изображения
// Возвращает маску сегментации, где каждый пиксель содержит номер кластера (0..K-1)
cv::Mat segment_kmeans(const cv::Mat& img, int K, int attempts = 3, 
                       cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0));

// Watershed сегментация
// markers - входная маска с маркерами (разные значения = разные регионы)
cv::Mat segment_watershed(const cv::Mat& img, const cv::Mat& markers);

// Визуализация сегментации: раскрашивает каждый сегмент случайным цветом
cv::Mat visualize_segmentation(const cv::Mat& segmentation_mask, 
                               const cv::Mat& original_img = cv::Mat(),
                               double overlay_alpha = 0.5);

// Визуализация отклонений от эталона: показывает ошибки сегментации
// predicted - предсказанная сегментация, ground_truth - эталонная
// Возвращает изображение с цветовой кодировкой ошибок:
// - Зеленый: правильно сегментированные пиксели
// - Красный: ложные срабатывания (FP)
// - Синий: пропущенные пиксели (FN)
cv::Mat visualize_segmentation_errors(const cv::Mat& predicted, 
                                      const cv::Mat& ground_truth,
                                      const cv::Mat& original_img = cv::Mat(),
                                      double overlay_alpha = 0.3);

// --- Lab04: Оценка качества сегментации (многоклассовая) ---

struct SegmentationMetrics {
    int num_classes{0};
    std::vector<int> class_ids;  // ID классов в сегментации
    
    // Для каждого класса
    std::vector<int> TP;  // True Positive для каждого класса
    std::vector<int> FP;  // False Positive для каждого класса
    std::vector<int> FN;  // False Negative для каждого класса
    
    // Метрики для каждого класса
    std::vector<double> Precision;  // Precision для каждого класса
    std::vector<double> Recall;     // Recall для каждого класса
    std::vector<double> IoU;        // IoU для каждого класса
    std::vector<double> F1;         // F1-score для каждого класса
    
    // Общие метрики
    double OverallAccuracy{0.0};    // Общая точность
    double MeanIoU{0.0};            // Средний IoU по всем классам
    double MeanF1{0.0};             // Средний F1 по всем классам
    
    // Вычисление метрик
    void compute();
};

// Вычисление метрик качества сегментации (для нескольких классов)
SegmentationMetrics calc_segmentation_metrics(const cv::Mat& predicted_segmentation,
                                              const cv::Mat& ground_truth_segmentation);

// --- Lab05: Детектирование объектов ---

// Структура для представления детектированного объекта
struct DetectedObject {
    cv::Rect bbox;           // Bounding box объекта
    cv::Point center;         // Центр объекта
    double confidence{0.0};  // Достоверность детектирования (0.0 - 1.0)
    int label{0};            // Метка объекта (опционально)
    int scaleLevel{0};        // Уровень масштаба, на котором был обнаружен объект
    cv::Mat mask;            // Маска объекта
    double area{0.0};         // Площадь объекта
    
    DetectedObject() = default;
    DetectedObject(const cv::Rect& box, double conf, int lbl = 0)
        : bbox(box), center(box.x + box.width/2, box.y + box.height/2),
          confidence(conf), label(lbl), scaleLevel(0), area(box.area()) {}
    
    DetectedObject(const cv::Rect& box, double conf, int level, const cv::Mat& m, int lbl = 0)
        : bbox(box), center(box.x + box.width/2, box.y + box.height/2),
          confidence(conf), label(lbl), scaleLevel(level), area(box.area()) {
        if (!m.empty()) {
            m.copyTo(mask);
        }
    }
};

// Локализация объектов интереса на изображении
// Использует анализ компонент связности и многомасштабный анализ
// min_area - минимальная площадь объекта для фильтрации шума
// max_area - максимальная площадь объекта
// scale_factors - факторы масштабирования для многомасштабного анализа
std::vector<DetectedObject> detect_objects(const cv::Mat& img,
                                           int min_area = 100,
                                           int max_area = 1000000,
                                           const std::vector<double>& scale_factors = {1.0, 0.5, 2.0});

// Оценка достоверности локализации объекта
// Вычисляет confidence score на основе размера, формы, контраста
double estimate_detection_confidence(const cv::Mat& img,
                                    const cv::Rect& bbox,
                                    const cv::Mat& binary_mask);

// Визуализация детектированных объектов
// Рисует bounding boxes на изображении с цветовой кодировкой по confidence
cv::Mat visualize_detections(const cv::Mat& img,
                            const std::vector<DetectedObject>& detections,
                            const cv::Scalar& color = cv::Scalar(0, 255, 0),
                            double thickness = 2.0);

// Визуализация сравнения детектированных объектов с эталоном
// Показывает правильные детекции (зеленый), ложные срабатывания (красный), пропущенные (синий)
cv::Mat visualize_detection_errors(const cv::Mat& img,
                                 const std::vector<DetectedObject>& predicted,
                                 const std::vector<DetectedObject>& ground_truth,
                                 double iou_threshold = 0.5);

// Метрики качества детектирования
struct DetectionMetrics {
    int TP{0};  // True Positive (правильно детектированные объекты)
    int FP{0};  // False Positive (ложные срабатывания)
    int FN{0};  // False Negative (пропущенные объекты)
    
    double Precision() const { return (TP + FP > 0) ? double(TP) / (TP + FP) : 0.0; }
    double Recall() const { return (TP + FN > 0) ? double(TP) / (TP + FN) : 0.0; }
    double F1() const {
        double prec = Precision();
        double rec = Recall();
        return (prec + rec > 0) ? 2.0 * prec * rec / (prec + rec) : 0.0;
    }
    
    // Средний IoU для правильно детектированных объектов
    double MeanIoU{0.0};
    
    // Детальные метрики для каждого предсказанного объекта
    std::vector<double> detection_ious;  // IoU для каждого TP детектирования
};

// Вычисление метрик качества детектирования
// iou_threshold - минимальный IoU для считания детектирования правильным
DetectionMetrics calc_detection_metrics(const std::vector<DetectedObject>& predicted,
                                       const std::vector<DetectedObject>& ground_truth,
                                       double iou_threshold = 0.5);

// Вычисление IoU (Intersection over Union) для двух bounding boxes
double calculate_iou(const cv::Rect& box1, const cv::Rect& box2);

// --- Lab06: Векторизация границ объектов ---

// Структура для представления векторного контура
struct VectorizedContour {
    std::vector<cv::Point> points;  // Точки контура
    double area{0.0};                // Площадь контура
    double perimeter{0.0};           // Периметр контура
    cv::Rect bbox;                   // Bounding box контура
    int object_id{0};                // ID объекта (опционально)
    
    VectorizedContour() = default;
    VectorizedContour(const std::vector<cv::Point>& pts) : points(pts) {
        if (!points.empty()) {
            area = cv::contourArea(points);
            perimeter = cv::arcLength(points, true);
            bbox = cv::boundingRect(points);
        }
    }
};

// Нахождение и векторизация границ объектов интереса
// binary_mask - бинарная маска объектов
// min_area - минимальная площадь контура для фильтрации
// epsilon - параметр упрощения контура (Douglas-Peucker)
// Возвращает вектор контуров объектов
std::vector<VectorizedContour> vectorize_object_boundaries(const cv::Mat& binary_mask,
                                                           double min_area = 100.0,
                                                           double epsilon = 2.0);

// Упрощение контура с помощью алгоритма Douglas-Peucker
std::vector<cv::Point> simplify_contour(const std::vector<cv::Point>& contour, double epsilon);

// Сохранение векторных контуров в текстовый файл
// Формат: каждая строка - один контур, точки разделены запятыми, контуры разделены пустой строкой
bool save_vectorized_contours(const std::string& filepath,
                              const std::vector<VectorizedContour>& contours);

// Загрузка векторных контуров из текстового файла
std::vector<VectorizedContour> load_vectorized_contours(const std::string& filepath);

// Визуализация векторных контуров на изображении
cv::Mat visualize_vectorized_contours(const cv::Mat& img,
                                     const std::vector<VectorizedContour>& contours,
                                     const cv::Scalar& color = cv::Scalar(0, 255, 0),
                                     int thickness = 2,
                                     bool fill = false);

// Визуализация сравнения векторных контуров с эталоном
// Показывает правильные контуры (зеленый), ложные (красный), пропущенные (синий)
cv::Mat visualize_vectorization_errors(const cv::Mat& img,
                                     const std::vector<VectorizedContour>& predicted,
                                     const std::vector<VectorizedContour>& ground_truth,
                                     double distance_threshold = 5.0);

// Метрики качества векторизации
struct VectorizationMetrics {
    int TP{0};  // True Positive (правильно векторизованные контуры)
    int FP{0};  // False Positive (ложные контуры)
    int FN{0};  // False Negative (пропущенные контуры)
    
    double Precision() const { return (TP + FP > 0) ? double(TP) / (TP + FP) : 0.0; }
    double Recall() const { return (TP + FN > 0) ? double(TP) / (TP + FN) : 0.0; }
    double F1() const {
        double prec = Precision();
        double rec = Recall();
        return (prec + rec > 0) ? 2.0 * prec * rec / (prec + rec) : 0.0;
    }
    
    // Среднее расстояние Хаусдорфа для правильно векторизованных контуров
    double MeanHausdorffDistance{0.0};
    
    // Среднее расстояние между контурами (average distance)
    double MeanContourDistance{0.0};
    
    // Средний IoU для контуров (на основе масок)
    double MeanIoU{0.0};
};

// Вычисление метрик качества векторизации
// distance_threshold - максимальное расстояние для считания контуров совпадающими (в пикселях)
VectorizationMetrics calc_vectorization_metrics(const std::vector<VectorizedContour>& predicted,
                                                const std::vector<VectorizedContour>& ground_truth,
                                                double distance_threshold = 5.0);

// Вычисление расстояния Хаусдорфа между двумя контурами
double hausdorff_distance(const std::vector<cv::Point>& contour1,
                         const std::vector<cv::Point>& contour2);

// Вычисление среднего расстояния между двумя контурами
double average_contour_distance(const std::vector<cv::Point>& contour1,
                                const std::vector<cv::Point>& contour2);

// Вычисление IoU для двух контуров (на основе масок)
double contour_iou(const std::vector<cv::Point>& contour1,
                  const std::vector<cv::Point>& contour2,
                  const cv::Size& image_size);

#endif  
