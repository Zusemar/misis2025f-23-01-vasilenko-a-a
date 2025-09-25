#include <opencv2/opencv.hpp>

#include <semcv/semcv.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
using namespace std::string_literals;

std::string strid_from_mat(const cv::Mat& img, const int n) {
    // Получаем параметры изображения
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
    
    if (!file.is_open()) {
        return file_paths; // Пустой вектор если файл не открылся
    }
    
    std::string filename;
    std::filesystem::path base_dir = path_lst.parent_path();
    
    while (std::getline(file, filename)) {
        // Убираем пробелы в начале и конце
        filename.erase(0, filename.find_first_not_of(" \t"));
        filename.erase(filename.find_last_not_of(" \t") + 1);
        
        if (filename.empty()) continue;
        
        // Создаем полный путь
        std::filesystem::path full_path = base_dir / filename;
        file_paths.push_back(full_path);
    }
    
    return file_paths;
}

cv::Mat generate_gray_bars_8u_768x30() {
    const int width = 768;
    const int height = 30;
    const int barWidth = 3; // 256 bars * 3px = 768px
    cv::Mat img(height, width, CV_8UC1);
    for (int x = 0; x < width; ++x) {
        int barIndex = x / barWidth; // 0..255
        if (barIndex < 0) barIndex = 0;
        if (barIndex > 255) barIndex = 255;
        unsigned char v = static_cast<unsigned char>(barIndex);
        for (int y = 0; y < height; ++y) {
            img.at<unsigned char>(y, x) = v;
        }
    }
    return img;
}

cv::Mat gamma_correction_8u(const cv::Mat& src, double gamma) {
    CV_Assert(src.type() == CV_8UC1);
    
    cv::Mat lut(1, 256, CV_8U);
    uchar* p = lut.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 1.0 / gamma) * 255.0);
    }
    
    cv::Mat dst;
    cv::LUT(src, lut, dst);
    return dst;
}