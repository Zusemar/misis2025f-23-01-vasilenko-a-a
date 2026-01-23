#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <set>

// Часть 1: Сегментация изображения
void part1_segmentation(const std::string& input_path, const std::string& output_dir, int K = 5) {
    // 1. Загружаем изображение
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }
    
    std::cout << "✓ Loaded image: " << input_path << " (" << img.cols << "x" << img.rows << ")" << std::endl;
    
    // 2. K-means сегментация
    std::cout << "\n=== K-means Segmentation (K=" << K << ") ===" << std::endl;
    cv::Mat kmeans_seg = segment_kmeans(img, K);
    
    std::string kmeans_path = output_dir + "/segmentation_kmeans.png";
    cv::imwrite(kmeans_path, kmeans_seg);
    std::cout << "✓ Saved K-means segmentation to " << kmeans_path << std::endl;
    
    // 3. Визуализация K-means сегментации
    cv::Mat kmeans_vis = visualize_segmentation(kmeans_seg, img, 0.5);
    std::string kmeans_vis_path = output_dir + "/segmentation_kmeans_visualized.png";
    cv::imwrite(kmeans_vis_path, kmeans_vis);
    std::cout << "✓ Saved K-means visualization to " << kmeans_vis_path << std::endl;
    
    // 4. Watershed сегментация (используем K-means для создания маркеров)
    std::cout << "\n=== Watershed Segmentation ===" << std::endl;
    
    // Создаем маркеры из K-means результата
    cv::Mat markers = cv::Mat::zeros(kmeans_seg.size(), CV_32SC1);
    cv::Mat gray = to_grayscale(img);
    
    // Преобразуем K-means результат в маркеры для watershed
    // Используем морфологические операции для улучшения маркеров
    cv::Mat kmeans_binary;
    cv::threshold(kmeans_seg, kmeans_binary, 0, 255, cv::THRESH_BINARY);
    
    // Находим контуры и создаем маркеры
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(kmeans_binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Создаем маркеры из контуров
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i) + 1), -1);
    }
    
    // Альтернативный подход: используем адаптивный порог для создания маркеров
    cv::Mat adaptive_thresh;
    cv::adaptiveThreshold(gray, adaptive_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                          cv::THRESH_BINARY_INV, 11, 2);
    
    // Морфологические операции для очистки
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(adaptive_thresh, adaptive_thresh, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(adaptive_thresh, adaptive_thresh, cv::MORPH_OPEN, kernel);
    
    // Находим маркеры (гарантированно передний план)
    cv::Mat sure_fg;
    cv::distanceTransform(adaptive_thresh, sure_fg, cv::DIST_L2, 5);
    cv::threshold(sure_fg, sure_fg, 0.7 * 255, 255, cv::THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8UC1);
    
    // Находим маркеры (гарантированно фон)
    cv::Mat sure_bg;
    cv::dilate(adaptive_thresh, sure_bg, kernel, cv::Point(-1, -1), 3);
    
    // Создаем маркеры: неизвестная область = 0, фон = 1, объекты = 2, 3, 4, ...
    markers = cv::Mat::zeros(adaptive_thresh.size(), CV_32SC1);
    markers.setTo(1, sure_bg);
    
    // Находим связанные компоненты в sure_fg и присваиваем им разные метки
    cv::Mat labels;
    int num_components = cv::connectedComponents(sure_fg, labels);
    for (int i = 1; i < num_components; ++i) {
        markers.setTo(i + 1, labels == i);
    }
    
    // Применяем watershed
    cv::Mat watershed_seg = segment_watershed(img, markers);
    
    std::string watershed_path = output_dir + "/segmentation_watershed.png";
    cv::imwrite(watershed_path, watershed_seg);
    std::cout << "✓ Saved Watershed segmentation to " << watershed_path << std::endl;
    
    // 5. Визуализация Watershed сегментации
    cv::Mat watershed_vis = visualize_segmentation(watershed_seg, img, 0.5);
    std::string watershed_vis_path = output_dir + "/segmentation_watershed_visualized.png";
    cv::imwrite(watershed_vis_path, watershed_vis);
    std::cout << "✓ Saved Watershed visualization to " << watershed_vis_path << std::endl;
    
    // Сохраняем исходное изображение для справки
    std::string original_path = output_dir + "/original.png";
    cv::imwrite(original_path, img);
    std::cout << "✓ Saved original image to " << original_path << std::endl;
}

// Часть 2: Генерация эталона сегментации
void part2_generate_ground_truth(const std::string& input_path, const std::string& output_dir) {
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }
    
    std::cout << "\n=== Generating Ground Truth Segmentation ===" << std::endl;
    
    cv::Mat gray = to_grayscale(img);
    
    // Создаем эталонную сегментацию на основе нескольких методов
    // Используем комбинацию методов для создания более точного эталона
    
    // Метод 1: K-means с оптимальным K
    cv::Mat gt_kmeans = segment_kmeans(img, 4);  // 4 класса: фон, скорлупа, ядро, границы
    
    // Метод 2: Адаптивная бинаризация + морфология
    cv::Mat adaptive;
    cv::adaptiveThreshold(gray, adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                          cv::THRESH_BINARY_INV, 15, 10);
    
    // Морфологические операции для выделения основных регионов
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(adaptive, adaptive, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(adaptive, adaptive, cv::MORPH_OPEN, kernel);
    
    // Создаем эталонную сегментацию: комбинируем результаты
    cv::Mat ground_truth = cv::Mat::zeros(img.size(), CV_8UC1);
    
    // Класс 1: Фон (темные области)
    cv::Mat dark_mask = gray < 50;
    ground_truth.setTo(1, dark_mask);
    
    // Класс 2: Скорлупа (средние значения)
    cv::Mat shell_mask = (gray >= 50) & (gray < 150);
    ground_truth.setTo(2, shell_mask);
    
    // Класс 3: Ядро (светлые области)
    cv::Mat kernel_mask = gray >= 150;
    ground_truth.setTo(3, kernel_mask);
    
    // Улучшаем эталон, используя результаты K-means
    // Нормализуем K-means результат к 3 классам
    cv::Mat kmeans_norm;
    cv::normalize(gt_kmeans, kmeans_norm, 1, 3, cv::NORM_MINMAX, CV_8UC1);
    
    // Комбинируем: используем K-means там, где уверены
    cv::Mat combined = ground_truth.clone();
    for (int y = 0; y < combined.rows; ++y) {
        for (int x = 0; x < combined.cols; ++x) {
            // Если K-means дает согласованный результат, используем его
            uchar kmeans_val = kmeans_norm.at<uchar>(y, x);
            if (kmeans_val > 0 && kmeans_val <= 3) {
                combined.at<uchar>(y, x) = kmeans_val;
            }
        }
    }
    
    std::string gt_path = output_dir + "/ground_truth_segmentation.png";
    cv::imwrite(gt_path, combined);
    std::cout << "✓ Saved ground truth segmentation to " << gt_path << std::endl;
    
    // Визуализация эталона
    cv::Mat gt_vis = visualize_segmentation(combined, img, 0.5);
    std::string gt_vis_path = output_dir + "/ground_truth_visualized.png";
    cv::imwrite(gt_vis_path, gt_vis);
    std::cout << "✓ Saved ground truth visualization to " << gt_vis_path << std::endl;
}

// Часть 3: Оценка качества сегментации
void part3_quality_assessment(const std::string& input_path,
                              const std::string& ground_truth_path,
                              const std::string& output_dir,
                              int K = 5) {
    cv::Mat img = cv::imread(input_path);
    cv::Mat ground_truth = cv::imread(ground_truth_path, cv::IMREAD_GRAYSCALE);
    
    if (img.empty() || ground_truth.empty()) {
        std::cerr << "Failed to load images" << std::endl;
        return;
    }
    
    std::cout << "\n=== Quality Assessment ===" << std::endl;
    
    // Выполняем сегментацию
    cv::Mat kmeans_seg = segment_kmeans(img, K);
    
    // Нормализуем K-means результат к диапазону классов в эталоне
    // Находим уникальные классы в эталоне
    std::set<uchar> gt_classes;
    for (int y = 0; y < ground_truth.rows; ++y) {
        for (int x = 0; x < ground_truth.cols; ++x) {
            uchar val = ground_truth.at<uchar>(y, x);
            if (val > 0) gt_classes.insert(val);
        }
    }
    
    int num_gt_classes = static_cast<int>(gt_classes.size());
    
    // Нормализуем предсказание к тому же диапазону классов
    cv::Mat predicted_norm;
    cv::normalize(kmeans_seg, predicted_norm, 1, num_gt_classes, cv::NORM_MINMAX, CV_8UC1);
    
    // Вычисляем метрики
    SegmentationMetrics metrics = calc_segmentation_metrics(predicted_norm, ground_truth);
    
    // Сохраняем метрики в CSV
    std::string csv_path = output_dir + "/segmentation_metrics.csv";
    std::ofstream csv(csv_path);
    csv << "Class_ID,TP,FP,FN,Precision,Recall,IoU,F1\n";
    
    for (int i = 0; i < metrics.num_classes; ++i) {
        csv << std::fixed << std::setprecision(4)
            << metrics.class_ids[i] << ","
            << metrics.TP[i] << ","
            << metrics.FP[i] << ","
            << metrics.FN[i] << ","
            << metrics.Precision[i] << ","
            << metrics.Recall[i] << ","
            << metrics.IoU[i] << ","
            << metrics.F1[i] << "\n";
    }
    
    csv << "\nOverall Metrics\n";
    csv << "OverallAccuracy,MeanIoU,MeanF1\n";
    csv << std::fixed << std::setprecision(4)
        << metrics.OverallAccuracy << ","
        << metrics.MeanIoU << ","
        << metrics.MeanF1 << "\n";
    csv.close();
    
    std::cout << "✓ Saved metrics to " << csv_path << std::endl;
    
    // Выводим метрики в консоль
    std::cout << "\n=== Segmentation Metrics ===" << std::endl;
    std::cout << "Number of classes: " << metrics.num_classes << std::endl;
    std::cout << "\nPer-class metrics:" << std::endl;
    for (int i = 0; i < metrics.num_classes; ++i) {
        std::cout << "  Class " << metrics.class_ids[i] << ":" << std::endl;
        std::cout << "    TP: " << metrics.TP[i] << ", FP: " << metrics.FP[i] 
                  << ", FN: " << metrics.FN[i] << std::endl;
        std::cout << "    Precision: " << metrics.Precision[i] 
                  << ", Recall: " << metrics.Recall[i] << std::endl;
        std::cout << "    IoU: " << metrics.IoU[i] 
                  << ", F1: " << metrics.F1[i] << std::endl;
    }
    std::cout << "\nOverall metrics:" << std::endl;
    std::cout << "  Overall Accuracy: " << metrics.OverallAccuracy << std::endl;
    std::cout << "  Mean IoU: " << metrics.MeanIoU << std::endl;
    std::cout << "  Mean F1: " << metrics.MeanF1 << std::endl;
    
    // Визуализация ошибок
    cv::Mat error_vis = visualize_segmentation_errors(predicted_norm, ground_truth, img, 0.3);
    std::string error_path = output_dir + "/segmentation_errors.png";
    cv::imwrite(error_path, error_vis);
    std::cout << "✓ Saved error visualization to " << error_path << std::endl;
    
    // Сохраняем предсказанную сегментацию
    std::string pred_path = output_dir + "/predicted_segmentation.png";
    cv::imwrite(pred_path, predicted_norm);
    std::cout << "✓ Saved predicted segmentation to " << pred_path << std::endl;
    
    // Визуализация предсказанной сегментации
    cv::Mat pred_vis = visualize_segmentation(predicted_norm, img, 0.5);
    std::string pred_vis_path = output_dir + "/predicted_segmentation_visualized.png";
    cv::imwrite(pred_vis_path, pred_vis);
    std::cout << "✓ Saved predicted visualization to " << pred_vis_path << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n"
                  << "  Part 1 (Segmentation): " << argv[0] << " part1 <input_image> <output_dir> [K]\n"
                  << "  Part 2 (Generate GT): " << argv[0] << " part2 <input_image> <output_dir>\n"
                  << "  Part 3 (Quality): " << argv[0] << " part3 <input_image> <ground_truth> <output_dir> [K]\n";
        return 1;
    }
    
    std::string mode = argv[1];
    
    if (mode == "part1") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " part1 <input_image> <output_dir> [K]" << std::endl;
            return 1;
        }
        int K = (argc >= 5) ? std::stoi(argv[4]) : 5;
        part1_segmentation(argv[2], argv[3], K);
    } else if (mode == "part2") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " part2 <input_image> <output_dir>" << std::endl;
            return 1;
        }
        part2_generate_ground_truth(argv[2], argv[3]);
    } else if (mode == "part3") {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0] << " part3 <input_image> <ground_truth> <output_dir> [K]" << std::endl;
            return 1;
        }
        int K = (argc >= 6) ? std::stoi(argv[5]) : 5;
        part3_quality_assessment(argv[2], argv[3], argv[4], K);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
    
    return 0;
}

