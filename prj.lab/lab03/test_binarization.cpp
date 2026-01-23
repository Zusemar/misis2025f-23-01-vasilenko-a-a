#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }
    
    std::string input_path = argv[1];
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return 1;
    }
    
    cv::Mat gray = to_grayscale(img);
    
    // Пробуем разные варианты бинаризации
    std::vector<std::pair<std::string, cv::Mat>> results;
    
    // 1. Обычная бинаризация с порогом 127
    cv::Mat bin1 = global_threshold(gray, 127.0);
    results.push_back({"binary_127.png", bin1});
    
    // 2. Инвертированная бинаризация с порогом 127
    cv::Mat bin2;
    cv::threshold(gray, bin2, 127, 255, cv::THRESH_BINARY_INV);
    results.push_back({"binary_inv_127.png", bin2});
    
    // 3. Метод Оцу (автоматический порог)
    cv::Mat bin3;
    double otsu_thresh = cv::threshold(gray, bin3, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    results.push_back({"binary_otsu.png", bin3});
    std::cout << "Otsu threshold: " << otsu_thresh << std::endl;
    
    // 4. Инвертированный Оцу
    cv::Mat bin4;
    double otsu_thresh_inv = cv::threshold(gray, bin4, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    results.push_back({"binary_otsu_inv.png", bin4});
    std::cout << "Otsu threshold (inv): " << otsu_thresh_inv << std::endl;
    
    // 5. Низкий порог (50)
    cv::Mat bin5 = global_threshold(gray, 50.0);
    results.push_back({"binary_50.png", bin5});
    
    // 6. Высокий порог (200)
    cv::Mat bin6 = global_threshold(gray, 200.0);
    results.push_back({"binary_200.png", bin6});
    
    // Сохраняем все результаты
    for (const auto& [filename, binary] : results) {
        std::string path = "prj.lab/lab03/output/" + filename;
        cv::imwrite(path, binary);
        std::cout << "✓ Saved: " << path << std::endl;
    }
    
    // Статистика по серому изображению
    cv::Scalar mean_val = cv::mean(gray);
    std::cout << "\nImage statistics:" << std::endl;
    std::cout << "  Mean brightness: " << mean_val[0] << std::endl;
    std::cout << "  Image size: " << gray.cols << "x" << gray.rows << std::endl;
    
    return 0;
}


