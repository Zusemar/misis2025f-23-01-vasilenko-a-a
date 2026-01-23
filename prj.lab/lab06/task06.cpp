#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <cmath>

// Часть 1: Нахождение и векторизация границ объектов интереса
void part1_vectorization(const std::string& input_path, const std::string& output_dir) {
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }
    
    std::cout << "✓ Loaded image: " << input_path << " (" << img.cols << "x" << img.rows << ")" << std::endl;
    
    std::cout << "\n=== Vectorization of Object Boundaries ===" << std::endl;
    
    // Преобразуем в серое изображение
    cv::Mat gray = to_grayscale(img);
    
    // Новый подход: используем комбинацию методов для лучшего выделения границ
    // 1. Улучшение контраста
    cv::Mat enhanced;
    cv::equalizeHist(gray, enhanced);
    
    // 2. Применяем фильтр для уменьшения шума
    cv::Mat filtered;
    cv::bilateralFilter(enhanced, filtered, 9, 75, 75);
    
    // 3. Бинаризация с несколькими методами и комбинирование
    cv::Mat binary1, binary2, binary3;
    
    // Адаптивная бинаризация
    cv::adaptiveThreshold(filtered, binary1, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY_INV, 15, 10);
    
    // Otsu бинаризация
    double otsu_thresh = cv::threshold(filtered, binary2, 0, 255, 
                                        cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    
    // Бинаризация на основе градиентов
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Mat grad_x_f, grad_y_f;
    cv::Sobel(filtered, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(filtered, grad_y, CV_16S, 0, 1, 3);
    grad_x.convertTo(grad_x_f, CV_32F);
    grad_y.convertTo(grad_y_f, CV_32F);
    cv::magnitude(grad_x_f, grad_y_f, grad_mag);
    grad_mag.convertTo(grad_mag, CV_8UC1);
    cv::threshold(grad_mag, binary3, 30, 255, cv::THRESH_BINARY);
    
    // Комбинируем методы: используем пересечение адаптивной и Otsu, объединяем с градиентами
    cv::Mat binary = (binary1 & binary2) | binary3;
    
    // Морфологические операции для очистки и соединения разрывов
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    
    // Векторизация границ объектов
    double min_area = 200.0;  // Минимальная площадь контура (увеличено для фильтрации шума)
    double epsilon = 1.5;     // Параметр упрощения контура (уменьшено для большей точности)
    
    std::vector<VectorizedContour> contours = vectorize_object_boundaries(binary, min_area, epsilon);
    
    std::cout << "✓ Vectorized " << contours.size() << " object boundaries" << std::endl;
    
    // Выводим информацию о векторизованных контурах
    for (size_t i = 0; i < contours.size(); ++i) {
        const auto& c = contours[i];
        std::cout << "  Contour " << i + 1 << ": " << c.points.size() << " points, "
                  << "area=" << std::fixed << std::setprecision(2) << c.area << ", "
                  << "perimeter=" << c.perimeter << std::endl;
    }
    
    // Визуализация векторизованных контуров
    cv::Mat vis = visualize_vectorized_contours(img, contours, cv::Scalar(0, 255, 0), 2, false);
    std::string vis_path = output_dir + "/vectorized_contours.png";
    cv::imwrite(vis_path, vis);
    std::cout << "✓ Saved vectorized contours visualization to " << vis_path << std::endl;
    
    // Визуализация с заполнением
    cv::Mat vis_filled = visualize_vectorized_contours(img, contours, cv::Scalar(0, 255, 0), 2, true);
    std::string vis_filled_path = output_dir + "/vectorized_contours_filled.png";
    cv::imwrite(vis_filled_path, vis_filled);
    std::cout << "✓ Saved filled vectorized contours to " << vis_filled_path << std::endl;
    
    // Сохраняем исходное изображение для справки
    std::string original_path = output_dir + "/original.png";
    cv::imwrite(original_path, img);
    std::cout << "✓ Saved original image to " << original_path << std::endl;
    
    // Сохраняем бинарную маску
    std::string binary_path = output_dir + "/binary_mask.png";
    cv::imwrite(binary_path, binary);
    std::cout << "✓ Saved binary mask to " << binary_path << std::endl;
}

// Часть 2: Генерация эталона векторного представления
void part2_generate_ground_truth(const std::string& input_path, const std::string& output_dir) {
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }
    
    std::cout << "\n=== Generating Ground Truth Vector Representation ===" << std::endl;
    
    cv::Mat gray = to_grayscale(img);
    
    // Создаем эталон на основе улучшенного метода
    // Используем более консервативные параметры для эталона
    
    // 1. Улучшение контраста
    cv::Mat enhanced;
    cv::equalizeHist(gray, enhanced);
    
    // 2. Фильтрация шума
    cv::Mat filtered;
    cv::bilateralFilter(enhanced, filtered, 9, 75, 75);
    
    // 3. Комбинация методов бинаризации
    cv::Mat binary1, binary2, binary3;
    
    // Адаптивная бинаризация с более строгими параметрами
    cv::adaptiveThreshold(filtered, binary1, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY_INV, 17, 8);
    
    // Otsu бинаризация
    double otsu_thresh = cv::threshold(filtered, binary2, 0, 255, 
                                       cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    
    // Градиентная бинаризация
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Mat grad_x_f, grad_y_f;
    cv::Sobel(filtered, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(filtered, grad_y, CV_16S, 0, 1, 3);
    grad_x.convertTo(grad_x_f, CV_32F);
    grad_y.convertTo(grad_y_f, CV_32F);
    cv::magnitude(grad_x_f, grad_y_f, grad_mag);
    grad_mag.convertTo(grad_mag, CV_8UC1);
    cv::threshold(grad_mag, binary3, 25, 255, cv::THRESH_BINARY);
    
    // Комбинируем: пересечение адаптивной и Otsu, объединяем с градиентами
    cv::Mat combined = (binary1 & binary2) | binary3;
    
    // Морфологические операции для очистки (более агрессивные для эталона)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(combined, combined, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);
    cv::morphologyEx(combined, combined, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);
    
    // Векторизация эталона с более строгими параметрами
    double min_area = 200.0;  // Минимальная площадь для эталона (увеличено)
    double epsilon = 1.0;      // Более точное упрощение для эталона (уменьшено)
    
    std::vector<VectorizedContour> ground_truth = vectorize_object_boundaries(combined, min_area, epsilon);
    
    std::cout << "✓ Generated " << ground_truth.size() << " ground truth contours" << std::endl;
    
    // Сохраняем эталон
    cv::Mat gt_vis = visualize_vectorized_contours(img, ground_truth, cv::Scalar(0, 255, 0), 3, false);
    std::string gt_path = output_dir + "/ground_truth_contours.png";
    cv::imwrite(gt_path, gt_vis);
    std::cout << "✓ Saved ground truth contours to " << gt_path << std::endl;
    
    // Сохраняем эталон в векторном формате
    std::string gt_vector_path = output_dir + "/ground_truth_contours.txt";
    if (save_vectorized_contours(gt_vector_path, ground_truth)) {
        std::cout << "✓ Saved ground truth contours (vector format) to " << gt_vector_path << std::endl;
    } else {
        std::cerr << "✗ Failed to save ground truth contours to " << gt_vector_path << std::endl;
    }
    
    // Визуализация эталона с заполнением
    cv::Mat gt_vis_filled = visualize_vectorized_contours(img, ground_truth, cv::Scalar(0, 255, 0), 3, true);
    std::string gt_filled_path = output_dir + "/ground_truth_contours_filled.png";
    cv::imwrite(gt_filled_path, gt_vis_filled);
    std::cout << "✓ Saved ground truth contours (filled) to " << gt_filled_path << std::endl;
}

// Часть 3: Оценка качества векторизации
void part3_quality_assessment(const std::string& input_path,
                              const std::string& ground_truth_path,
                              const std::string& output_dir) {
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }
    
    // Загружаем эталон из векторного файла
    std::vector<VectorizedContour> ground_truth = load_vectorized_contours(ground_truth_path);
    
    if (ground_truth.empty()) {
        std::cerr << "Warning: Could not load ground truth from " << ground_truth_path 
                  << ". Generating new ground truth." << std::endl;
        // Генерируем эталон на лету
        part2_generate_ground_truth(input_path, output_dir);
        std::string new_gt_path = output_dir + "/ground_truth_contours.txt";
        ground_truth = load_vectorized_contours(new_gt_path);
    }
    
    std::cout << "\n=== Quality Assessment ===" << std::endl;
    std::cout << "Ground truth contours: " << ground_truth.size() << std::endl;
    
    // Выполняем векторизацию с улучшенным методом
    cv::Mat gray = to_grayscale(img);
    
    // Улучшение контраста
    cv::Mat enhanced;
    cv::equalizeHist(gray, enhanced);
    
    // Фильтрация шума
    cv::Mat filtered;
    cv::bilateralFilter(enhanced, filtered, 9, 75, 75);
    
    // Комбинация методов бинаризации
    cv::Mat binary1, binary2, binary3;
    cv::adaptiveThreshold(filtered, binary1, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY_INV, 15, 10);
    double otsu_thresh = cv::threshold(filtered, binary2, 0, 255, 
                                       cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Mat grad_x_f, grad_y_f;
    cv::Sobel(filtered, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(filtered, grad_y, CV_16S, 0, 1, 3);
    grad_x.convertTo(grad_x_f, CV_32F);
    grad_y.convertTo(grad_y_f, CV_32F);
    cv::magnitude(grad_x_f, grad_y_f, grad_mag);
    grad_mag.convertTo(grad_mag, CV_8UC1);
    cv::threshold(grad_mag, binary3, 30, 255, cv::THRESH_BINARY);
    
    cv::Mat binary = (binary1 & binary2) | binary3;
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    
    double min_area = 200.0;
    double epsilon = 1.5;
    
    std::vector<VectorizedContour> predicted = vectorize_object_boundaries(binary, min_area, epsilon);
    std::cout << "Predicted contours: " << predicted.size() << std::endl;
    
    // Вычисляем метрики
    double distance_threshold = 5.0;  // Порог расстояния для считания контуров совпадающими
    VectorizationMetrics metrics = calc_vectorization_metrics(predicted, ground_truth, distance_threshold);
    
    // Сохраняем метрики в CSV
    std::string csv_path = output_dir + "/vectorization_metrics.csv";
    std::ofstream csv(csv_path);
    csv << "Metric,Value\n";
    csv << "TP," << metrics.TP << "\n";
    csv << "FP," << metrics.FP << "\n";
    csv << "FN," << metrics.FN << "\n";
    csv << "Precision," << std::fixed << std::setprecision(4) << metrics.Precision() << "\n";
    csv << "Recall," << metrics.Recall() << "\n";
    csv << "F1," << metrics.F1() << "\n";
    csv << "MeanHausdorffDistance," << metrics.MeanHausdorffDistance << "\n";
    csv << "MeanContourDistance," << metrics.MeanContourDistance << "\n";
    csv << "MeanIoU," << metrics.MeanIoU << "\n";
    csv.close();
    
    std::cout << "✓ Saved metrics to " << csv_path << std::endl;
    
    // Выводим метрики в консоль
    std::cout << "\n=== Vectorization Metrics ===" << std::endl;
    std::cout << "TP: " << metrics.TP << ", FP: " << metrics.FP << ", FN: " << metrics.FN << std::endl;
    std::cout << "Precision: " << std::fixed << std::setprecision(4) << metrics.Precision() << std::endl;
    std::cout << "Recall: " << metrics.Recall() << std::endl;
    std::cout << "F1: " << metrics.F1() << std::endl;
    std::cout << "Mean Hausdorff Distance: " << metrics.MeanHausdorffDistance << std::endl;
    std::cout << "Mean Contour Distance: " << metrics.MeanContourDistance << std::endl;
    std::cout << "Mean IoU: " << metrics.MeanIoU << std::endl;
    
    // Визуализация ошибок векторизации
    cv::Mat error_vis = visualize_vectorization_errors(img, predicted, ground_truth, distance_threshold);
    std::string error_path = output_dir + "/vectorization_errors.png";
    cv::imwrite(error_path, error_vis);
    std::cout << "✓ Saved error visualization to " << error_path << std::endl;
    
    // Визуализация предсказанных контуров
    cv::Mat pred_vis = visualize_vectorized_contours(img, predicted, cv::Scalar(0, 255, 0), 2, false);
    std::string pred_path = output_dir + "/predicted_contours.png";
    cv::imwrite(pred_path, pred_vis);
    std::cout << "✓ Saved predicted contours to " << pred_path << std::endl;
    
    // Визуализация эталона
    cv::Mat gt_vis = visualize_vectorized_contours(img, ground_truth, cv::Scalar(0, 255, 0), 3, false);
    std::string gt_vis_path = output_dir + "/ground_truth_visualized.png";
    cv::imwrite(gt_vis_path, gt_vis);
    std::cout << "✓ Saved ground truth visualization to " << gt_vis_path << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n"
                  << "  Part 1 (Vectorization): " << argv[0] << " part1 <input_image> <output_dir>\n"
                  << "  Part 2 (Generate GT): " << argv[0] << " part2 <input_image> <output_dir>\n"
                  << "  Part 3 (Quality): " << argv[0] << " part3 <input_image> <ground_truth_txt> <output_dir>\n";
        return 1;
    }
    
    std::string mode = argv[1];
    
    if (mode == "part1") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " part1 <input_image> <output_dir>" << std::endl;
            return 1;
        }
        part1_vectorization(argv[2], argv[3]);
    } else if (mode == "part2") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " part2 <input_image> <output_dir>" << std::endl;
            return 1;
        }
        part2_generate_ground_truth(argv[2], argv[3]);
    } else if (mode == "part3") {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0] << " part3 <input_image> <ground_truth_txt> <output_dir>" << std::endl;
            return 1;
        }
        part3_quality_assessment(argv[2], argv[3], argv[4]);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
    
    return 0;
}

