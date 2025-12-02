#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

// Часть 1: базовая бинаризация и визуализация
void part1_binarization(const std::string& input_path, const std::string& output_dir, double threshold = 127.0) {
    // 1. Загружаем изображение
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }
    
    // 2. Получаем серое одноканальное изображение
    cv::Mat gray = to_grayscale(img);
    
    // 3. Строим и визуализируем гистограмму
    cv::Mat hist_img = draw_histogram_8u(gray);
    std::string hist_path = output_dir + "/histogram.png";
    cv::imwrite(hist_path, hist_img);
    std::cout << "✓ Saved histogram to " << hist_path << std::endl;
    
    // 4. Простая глобальная бинаризация
    cv::Mat binary = global_threshold(gray, threshold);
    std::string binary_path = output_dir + "/binary_mask.png";
    cv::imwrite(binary_path, binary);
    std::cout << "✓ Saved binary mask to " << binary_path << std::endl;
    
    // 5. Визуализация маски: наложение полупрозрачной маски на исходное изображение
    cv::Mat overlay = overlay_mask(img, binary, cv::Scalar(0, 255, 0), 0.5); // зелёная маска, 50% прозрачности
    std::string overlay_path = output_dir + "/overlay.png";
    cv::imwrite(overlay_path, overlay);
    std::cout << "✓ Saved overlay to " << overlay_path << std::endl;
    
    // Сохраняем также серое изображение для справки
    std::string gray_path = output_dir + "/grayscale.png";
    cv::imwrite(gray_path, gray);
    std::cout << "✓ Saved grayscale to " << gray_path << std::endl;
}

// Часть 2: оценка качества бинаризации
void part2_quality_assessment(const std::string& input_path, 
                              const std::string& ground_truth_path,
                              const std::string& output_dir) {
    // Загружаем изображение и эталонную маску
    cv::Mat img = cv::imread(input_path);
    cv::Mat gt_mask = cv::imread(ground_truth_path, cv::IMREAD_GRAYSCALE);
    
    if (img.empty() || gt_mask.empty()) {
        std::cerr << "Failed to load images" << std::endl;
        return;
    }
    
    cv::Mat gray = to_grayscale(img);
    
    // Анализ зависимости качества от порога
    std::vector<double> thresholds;
    std::vector<BinaryClassificationMetrics> metrics_list;
    
    // Перебираем пороги от 0 до 255 с шагом 5
    for (int t = 0; t <= 255; t += 5) {
        cv::Mat binary = global_threshold(gray, static_cast<double>(t));
        BinaryClassificationMetrics m = calc_binary_metrics(binary, gt_mask);
        thresholds.push_back(t);
        metrics_list.push_back(m);
    }
    
    // Сохраняем результаты в CSV
    std::string csv_path = output_dir + "/threshold_analysis.csv";
    std::ofstream csv(csv_path);
    csv << "threshold,TP,FP,FN,TN,TPR,FPR,Precision,IoU,Accuracy\n";
    
    for (size_t i = 0; i < thresholds.size(); ++i) {
        const auto& m = metrics_list[i];
        csv << std::fixed << std::setprecision(2)
            << thresholds[i] << ","
            << m.TP << "," << m.FP << "," << m.FN << "," << m.TN << ","
            << m.TPR() << "," << m.FPR() << ","
            << m.Precision() << "," << m.IoU() << ","
            << m.Accuracy() << "\n";
    }
    csv.close();
    std::cout << "✓ Saved threshold analysis to " << csv_path << std::endl;
    
    // Находим оптимальный порог по IoU
    double best_iou = 0.0;
    int best_threshold_idx = 0;
    for (size_t i = 0; i < metrics_list.size(); ++i) {
        if (metrics_list[i].IoU() > best_iou) {
            best_iou = metrics_list[i].IoU();
            best_threshold_idx = i;
        }
    }
    
    double best_threshold = thresholds[best_threshold_idx];
    std::cout << "✓ Best threshold: " << best_threshold 
              << " (IoU = " << best_iou << ")" << std::endl;
    
    // Создаём бинаризацию с оптимальным порогом
    cv::Mat best_binary = global_threshold(gray, best_threshold);
    std::string best_binary_path = output_dir + "/best_binary.png";
    cv::imwrite(best_binary_path, best_binary);
    
    // Визуализация с оптимальной маской
    cv::Mat best_overlay = overlay_mask(img, best_binary, cv::Scalar(0, 255, 0), 0.5);
    std::string best_overlay_path = output_dir + "/best_overlay.png";
    cv::imwrite(best_overlay_path, best_overlay);
    
    // Выводим метрики для оптимального порога
    const auto& best_metrics = metrics_list[best_threshold_idx];
    std::cout << "\n=== Best Threshold Metrics ===" << std::endl;
    std::cout << "Threshold: " << best_threshold << std::endl;
    std::cout << "TP: " << best_metrics.TP << ", FP: " << best_metrics.FP 
              << ", FN: " << best_metrics.FN << ", TN: " << best_metrics.TN << std::endl;
    std::cout << "TPR (Recall): " << best_metrics.TPR() << std::endl;
    std::cout << "FPR: " << best_metrics.FPR() << std::endl;
    std::cout << "Precision: " << best_metrics.Precision() << std::endl;
    std::cout << "IoU: " << best_metrics.IoU() << std::endl;
    std::cout << "Accuracy: " << best_metrics.Accuracy() << std::endl;
}

// Генерация эталонных масок (для тестирования)
void generate_ground_truth_masks(const std::string& input_path, const std::string& output_dir) {
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }
    
    cv::Mat gray = to_grayscale(img);
    
    // Вариант 1: простая пороговая маска (можно использовать Paint.net для ручной правки)
    cv::Mat gt1 = global_threshold(gray, 127.0);
    std::string gt1_path = output_dir + "/ground_truth_127.png";
    cv::imwrite(gt1_path, gt1);
    
    // Вариант 2: адаптивный порог (Otsu)
    cv::Mat gt2;
    cv::threshold(gray, gt2, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    std::string gt2_path = output_dir + "/ground_truth_otsu.png";
    cv::imwrite(gt2_path, gt2);
    
    // Вариант 3: адаптивная бинаризация
    cv::Mat gt3;
    cv::adaptiveThreshold(gray, gt3, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                          cv::THRESH_BINARY, 11, 2);
    std::string gt3_path = output_dir + "/ground_truth_adaptive.png";
    cv::imwrite(gt3_path, gt3);
    
    std::cout << "✓ Generated ground truth masks:" << std::endl;
    std::cout << "  - " << gt1_path << " (threshold 127)" << std::endl;
    std::cout << "  - " << gt2_path << " (Otsu)" << std::endl;
    std::cout << "  - " << gt3_path << " (adaptive)" << std::endl;
    std::cout << "  You can manually edit these in Paint.net to create reference masks" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n"
                  << "  Part 1: " << argv[0] << " part1 <input_image> <output_dir> [threshold]\n"
                  << "  Part 2: " << argv[0] << " part2 <input_image> <ground_truth_mask> <output_dir>\n"
                  << "  Generate GT: " << argv[0] << " gen_gt <input_image> <output_dir>\n";
        return 1;
    }
    
    std::string mode = argv[1];
    
    if (mode == "part1") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " part1 <input_image> <output_dir> [threshold]" << std::endl;
            return 1;
        }
        double threshold = (argc >= 5) ? std::stod(argv[4]) : 127.0;
        part1_binarization(argv[2], argv[3], threshold);
    } else if (mode == "part2") {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0] << " part2 <input_image> <ground_truth_mask> <output_dir>" << std::endl;
            return 1;
        }
        part2_quality_assessment(argv[2], argv[3], argv[4]);
    } else if (mode == "gen_gt") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " gen_gt <input_image> <output_dir>" << std::endl;
            return 1;
        }
        generate_ground_truth_masks(argv[2], argv[3]);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
    
    return 0;
}

