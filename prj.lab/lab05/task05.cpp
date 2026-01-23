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
#include <algorithm>
#include <map>
#include <chrono>

using namespace std::chrono;

// Класс для эталона детектирования
class GroundTruth {
private:
    std::vector<DetectedObject> objects;
    std::string name;
    
public:
    GroundTruth(const std::string& n) : name(n) {}
    
    void addObject(const DetectedObject& obj) {
        objects.push_back(obj);
    }
    
    const std::vector<DetectedObject>& getObjects() const {
        return objects;
    }
    
    void saveToFile(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        file << "Ground Truth: " << name << std::endl;
        file << "Object Count: " << objects.size() << std::endl;
        file << "x,y,width,height,confidence" << std::endl;
        
        for (size_t i = 0; i < objects.size(); i++) {
            const auto& obj = objects[i];
            file << obj.bbox.x << "," << obj.bbox.y << ","
                 << obj.bbox.width << "," << obj.bbox.height << ","
                 << std::fixed << std::setprecision(3) << obj.confidence << std::endl;
        }
        file.close();
    }
    
    void loadFromFile(const std::string& filename) {
        objects.clear();
        std::ifstream file(filename);
        if (!file.is_open()) return;
        
        std::string line;
        std::getline(file, line);  // Пропускаем "Ground Truth: ..."
        std::getline(file, line);  // Пропускаем "Object Count: ..."
        std::getline(file, line);  // Пропускаем заголовок CSV
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> values;
            
            while (std::getline(iss, token, ',')) {
                values.push_back(token);
            }
            
            if (values.size() >= 4) {
                try {
                    int x = std::stoi(values[0]);
                    int y = std::stoi(values[1]);
                    int w = std::stoi(values[2]);
                    int h = std::stoi(values[3]);
                    double conf = (values.size() >= 5) ? std::stod(values[4]) : 0.9;
                    
                    cv::Rect bbox(x, y, w, h);
                    objects.emplace_back(bbox, conf);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to parse line: " << line << std::endl;
                }
            }
        }
        file.close();
    }
};

// Часть 1: Локализация объектов интереса
void part1_localization(const std::string& input_path, const std::string& output_dir) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "PART 1: Object Localization" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Error: Failed to load image: " << input_path << std::endl;
        return;
    }
    
    std::cout << "Image loaded: " << input_path << std::endl;
    std::cout << "Size: " << img.cols << "x" << img.rows << " pixels" << std::endl;
    
    std::filesystem::create_directories(output_dir);
    
    // Параметры детектирования
    int min_area = 150;  // Увеличена минимальная площадь для более строгой фильтрации
    int max_area = static_cast<int>(img.rows * img.cols * 0.5);
    std::vector<double> scale_factors = {2.0, 1.5, 1.0, 0.75, 0.5};
    
    std::cout << "\nDetection parameters:" << std::endl;
    std::cout << "  Min area: " << min_area << " pixels" << std::endl;
    std::cout << "  Max area: " << max_area << " pixels" << std::endl;
    std::cout << "  Scale factors: ";
    for (double s : scale_factors) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    
    // Выполняем детекцию
    std::cout << "\nPerforming multi-scale object detection..." << std::endl;
    auto start = high_resolution_clock::now();
    std::vector<DetectedObject> detections = detect_objects(img, min_area, max_area, scale_factors);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    
    std::cout << "Detection time: " << duration.count() << " ms" << std::endl;
    std::cout << "Objects detected: " << detections.size() << std::endl;
    
    // Вывод информации о детекциях
    if (!detections.empty()) {
        std::cout << "\nDetected objects:" << std::endl;
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            std::cout << "  [" << (i + 1) << "] bbox=(" << det.bbox.x << "," << det.bbox.y 
                      << "," << det.bbox.width << "," << det.bbox.height 
                      << ") confidence=" << std::fixed << std::setprecision(3) 
                      << det.confidence << " scale=" << det.scaleLevel << std::endl;
        }
    }
    
    // Анализ достоверности
    if (!detections.empty()) {
        double sum_conf = 0.0;
        double min_conf = 1.0;
        double max_conf = 0.0;
        
        for (const auto& det : detections) {
            sum_conf += det.confidence;
            min_conf = std::min(min_conf, det.confidence);
            max_conf = std::max(max_conf, det.confidence);
        }
        
        double avg_conf = sum_conf / detections.size();
        
        std::cout << "\nConfidence statistics:" << std::endl;
        std::cout << "  Average: " << std::fixed << std::setprecision(3) << avg_conf << std::endl;
        std::cout << "  Min: " << min_conf << std::endl;
        std::cout << "  Max: " << max_conf << std::endl;
    }
    
    // Визуализация детекций
    cv::Mat detections_vis = visualize_detections(img, detections);
    std::string detections_path = output_dir + "/detections.png";
    cv::imwrite(detections_path, detections_vis);
    std::cout << "\nSaved: " << detections_path << std::endl;
    
    // Сохранение исходного изображения
    std::string original_path = output_dir + "/original.png";
    cv::imwrite(original_path, img);
    std::cout << "Saved: " << original_path << std::endl;
    
    // Анализ компонент связности
    std::cout << "\nAnalyzing connected components..." << std::endl;
    cv::Mat gray = to_grayscale(img);
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY_INV, 11, 2);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    
    cv::Mat labels, stats, centroids;
    int num_components = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
    
    std::cout << "Total connected components: " << num_components - 1 << " (excluding background)" << std::endl;
    
    int valid_components = 0;
    for (int i = 1; i < num_components; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_area && area <= max_area) {
            valid_components++;
        }
    }
    std::cout << "Valid components (after filtering): " << valid_components << std::endl;
    
    // Визуализация компонент связности
    cv::Mat components_vis = cv::Mat::zeros(img.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors(num_components);
    for (int i = 0; i < num_components; ++i) {
        colors[i] = cv::Vec3b((i * 137) % 256, (i * 199) % 256, (i * 73) % 256);
    }
    
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            int label = labels.at<int>(y, x);
            if (label > 0) {
                components_vis.at<cv::Vec3b>(y, x) = colors[label];
            }
        }
    }
    
    std::string components_path = output_dir + "/connected_components.png";
    cv::imwrite(components_path, components_vis);
    std::cout << "Saved: " << components_path << std::endl;
    
    // Сохранение бинарной маски
    std::string binary_path = output_dir + "/binary_mask.png";
    cv::imwrite(binary_path, binary);
    std::cout << "Saved: " << binary_path << std::endl;
    
    // Сохранение детекций в текстовый формат
    std::string detections_txt_path = output_dir + "/detections.txt";
    std::ofstream det_file(detections_txt_path);
    if (det_file.is_open()) {
        det_file << "x,y,width,height,confidence\n";
        for (const auto& det : detections) {
            det_file << det.bbox.x << "," << det.bbox.y << ","
                     << det.bbox.width << "," << det.bbox.height << ","
                     << std::fixed << std::setprecision(3) << det.confidence << "\n";
        }
        det_file.close();
        std::cout << "Saved: " << detections_txt_path << std::endl;
    }
    
    std::cout << "\nPart 1 completed successfully!" << std::endl;
}

// Часть 2: Генерация эталона детектирования
void part2_generate_ground_truth(const std::string& input_path, const std::string& output_dir) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "PART 2: Ground Truth Generation" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Error: Failed to load image: " << input_path << std::endl;
        return;
    }
    
    std::cout << "Image loaded: " << input_path << std::endl;
    std::cout << "Size: " << img.cols << "x" << img.rows << " pixels" << std::endl;
    
    std::filesystem::create_directories(output_dir);
    
    GroundTruth gt("Ground Truth Detections");
    
    cv::Mat gray = to_grayscale(img);
    
    // Создаем эталон на основе комбинации методов
    cv::Mat binary1, binary2;
    cv::adaptiveThreshold(gray, binary1, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY_INV, 15, 5);
    cv::threshold(gray, binary2, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    
    cv::Mat combined = binary1 & binary2;
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(combined, combined, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(combined, combined, cv::MORPH_OPEN, kernel);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combined, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::cout << "Found " << contours.size() << " contours" << std::endl;
    
    int min_area_threshold = 150;  // Увеличена минимальная площадь для эталона
    
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area >= min_area_threshold) {
            cv::Rect bbox = cv::boundingRect(contour);
            
            double aspect_ratio = (bbox.height > 0) ? double(bbox.width) / bbox.height : 0.0;
            if (aspect_ratio < 0.1 || aspect_ratio > 10.0) continue;
            
            if (bbox.width < 10 || bbox.height < 10) continue;
            
            cv::Mat obj_mask = cv::Mat::zeros(img.size(), CV_8UC1);
            std::vector<std::vector<cv::Point>> single_contour = {contour};
            cv::drawContours(obj_mask, single_contour, -1, cv::Scalar(255), -1);
            
            double confidence = estimate_detection_confidence(img, bbox, obj_mask);
            confidence = std::max(confidence, 0.85);
            
            DetectedObject obj(bbox, confidence, 0, obj_mask);
            gt.addObject(obj);
        }
    }
    
    std::cout << "Generated " << gt.getObjects().size() << " ground truth detections" << std::endl;
    
    // Сохраняем эталон
    std::string gt_path = output_dir + "/ground_truth_detections.txt";
    gt.saveToFile(gt_path);
    std::cout << "Saved: " << gt_path << std::endl;
    
    // Визуализация эталона
    cv::Mat gt_vis = visualize_detections(img, gt.getObjects(), cv::Scalar(0, 255, 0), 3);
    std::string gt_img_path = output_dir + "/ground_truth_detections.png";
    cv::imwrite(gt_img_path, gt_vis);
    std::cout << "Saved: " << gt_img_path << std::endl;
    
    // Сохранение бинарной маски
    std::string mask_path = output_dir + "/ground_truth_mask.png";
    cv::imwrite(mask_path, combined);
    std::cout << "Saved: " << mask_path << std::endl;
    
    std::cout << "\nPart 2 completed successfully!" << std::endl;
}

// Часть 3: Оценка качества детектирования
void part3_quality_assessment(const std::string& input_path,
                              const std::string& ground_truth_path,
                              const std::string& output_dir) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "PART 3: Quality Assessment" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Error: Failed to load image: " << input_path << std::endl;
        return;
    }
    
    std::filesystem::create_directories(output_dir);
    
    // Загружаем эталон
    GroundTruth gt("Ground Truth");
    gt.loadFromFile(ground_truth_path);
    
    if (gt.getObjects().empty()) {
        std::cerr << "Warning: Could not load ground truth. Generating new..." << std::endl;
        part2_generate_ground_truth(input_path, output_dir);
        std::string new_gt_path = output_dir + "/ground_truth_detections.txt";
        gt.loadFromFile(new_gt_path);
    }
    
    std::cout << "Ground truth loaded: " << gt.getObjects().size() << " objects" << std::endl;
    
    // Выполняем детектирование
    int min_area = 150;  // Увеличена минимальная площадь для более строгой фильтрации
    int max_area = static_cast<int>(img.rows * img.cols * 0.5);
    std::vector<double> scale_factors = {2.0, 1.5, 1.0, 0.75, 0.5};
    
    std::cout << "\nPerforming object detection..." << std::endl;
    std::vector<DetectedObject> predicted = detect_objects(img, min_area, max_area, scale_factors);
    std::cout << "Predicted: " << predicted.size() << " objects" << std::endl;
    
    // Вычисляем метрики
    double iou_threshold = 0.5;
    DetectionMetrics metrics = calc_detection_metrics(predicted, gt.getObjects(), iou_threshold);
    
    // Вывод метрик
    std::cout << "\nDetection Metrics:" << std::endl;
    std::cout << "  TP (True Positives): " << metrics.TP << std::endl;
    std::cout << "  FP (False Positives): " << metrics.FP << std::endl;
    std::cout << "  FN (False Negatives): " << metrics.FN << std::endl;
    std::cout << "  Precision: " << std::fixed << std::setprecision(4) 
              << metrics.Precision() << std::endl;
    std::cout << "  Recall: " << metrics.Recall() << std::endl;
    std::cout << "  F1-Score: " << metrics.F1() << std::endl;
    std::cout << "  Mean IoU: " << metrics.MeanIoU << std::endl;
    
    // Сохранение метрик в CSV
    std::string csv_path = output_dir + "/detection_metrics.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "Metric,Value\n";
        csv << "TP," << metrics.TP << "\n";
        csv << "FP," << metrics.FP << "\n";
        csv << "FN," << metrics.FN << "\n";
        csv << "Precision," << std::fixed << std::setprecision(4) << metrics.Precision() << "\n";
        csv << "Recall," << metrics.Recall() << "\n";
        csv << "F1," << metrics.F1() << "\n";
        csv << "MeanIoU," << metrics.MeanIoU << "\n";
        csv << "IoU_Threshold," << iou_threshold << "\n";
        csv.close();
        std::cout << "\nSaved: " << csv_path << std::endl;
    }
    
    // Визуализация ошибок детектирования
    cv::Mat error_vis = visualize_detection_errors(img, predicted, gt.getObjects(), iou_threshold);
    std::string error_path = output_dir + "/detection_errors.png";
    cv::imwrite(error_path, error_vis);
    std::cout << "Saved: " << error_path << std::endl;
    
    // Визуализация предсказанных детекций
    cv::Mat pred_vis = visualize_detections(img, predicted);
    std::string pred_path = output_dir + "/predicted_detections.png";
    cv::imwrite(pred_path, pred_vis);
    std::cout << "Saved: " << pred_path << std::endl;
    
    // Визуализация эталона
    cv::Mat gt_vis = visualize_detections(img, gt.getObjects(), cv::Scalar(0, 255, 0), 3);
    std::string gt_vis_path = output_dir + "/ground_truth_visualized.png";
    cv::imwrite(gt_vis_path, gt_vis);
    std::cout << "Saved: " << gt_vis_path << std::endl;
    
    // Комбинированная визуализация
    cv::Mat combined_vis = img.clone();
    if (combined_vis.channels() == 1) {
        cv::cvtColor(combined_vis, combined_vis, cv::COLOR_GRAY2BGR);
    }
    
    for (const auto& gt_obj : gt.getObjects()) {
        cv::rectangle(combined_vis, gt_obj.bbox, cv::Scalar(0, 255, 0), 2);
    }
    
    for (const auto& pred : predicted) {
        cv::rectangle(combined_vis, pred.bbox, cv::Scalar(255, 0, 0), 2);
    }
    
    std::string combined_path = output_dir + "/combined_detections.png";
    cv::imwrite(combined_path, combined_vis);
    std::cout << "Saved: " << combined_path << std::endl;
    
    std::cout << "\nPart 3 completed successfully!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n"
                  << "  Part 1 (Localization): " << argv[0] << " part1 <input_image> <output_dir>\n"
                  << "  Part 2 (Generate GT): " << argv[0] << " part2 <input_image> <output_dir>\n"
                  << "  Part 3 (Quality): " << argv[0] << " part3 <input_image> <ground_truth_txt> <output_dir>\n";
        return 1;
    }
    
    std::string mode = argv[1];
    
    try {
        if (mode == "part1") {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0] << " part1 <input_image> <output_dir>" << std::endl;
                return 1;
            }
            part1_localization(argv[2], argv[3]);
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
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
