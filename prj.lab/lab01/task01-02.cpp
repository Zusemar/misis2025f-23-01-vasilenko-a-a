#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <output.png>" << std::endl;
        return -1;
    }
    
    // Генерируем базовое изображение
    cv::Mat base_image = generate_gray_bars_8u_768x30();
    
    // Значения гамма для применения
    std::vector<double> gammas = {1.0, 1.8, 2.0, 2.2, 2.4, 2.6};
    std::vector<cv::Mat> images;
    std::vector<std::string> labels = {"γ=1.0", "γ=1.8", "γ=2.0", "γ=2.2", "γ=2.4", "γ=2.6"};
    
    // Создаем все варианты изображений
    for (double gamma : gammas) {
        images.push_back(gamma_correction_8u(base_image, gamma));
    }
    
    // Создаем вертикальный коллаж
    int total_height = 0;
    for (const auto& img : images) total_height += img.rows;
    
    cv::Mat collage(total_height, base_image.cols, CV_8UC1);
    
    int y = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        images[i].copyTo(collage(cv::Rect(0, y, images[i].cols, images[i].rows)));
        
        // Добавляем подпись
        cv::putText(collage, labels[i], cv::Point(10, y + 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, 255, 1);
        y += images[i].rows;
    }
    
    // Сохраняем результат
    if (cv::imwrite(argv[1], collage)) {
        std::cout << "Collage saved: " << argv[1] << std::endl;
        return 0;
    } else {
        std::cerr << "Error saving file!" << std::endl;
        return -1;
    }
}