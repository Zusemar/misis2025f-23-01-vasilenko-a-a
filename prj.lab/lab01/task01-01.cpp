// task01-01: validate that file names match raster descriptors
#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <semcv/semcv.hpp>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: task01-01 <path/to/task01.lst>" << std::endl;
    return 1;
  }

  const std::filesystem::path lst_path = argv[1];
  const auto files = get_list_of_file_paths(lst_path);
  for (const auto& file_path : files) {
    cv::Mat img = cv::imread(file_path.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
      std::cout << file_path.string() << "\t" << "bad, should be " << "unreadable" << std::endl;
      continue;
    }
    const std::string expected = strid_from_mat(img, 4);
    const std::string stem = file_path.stem().string();
    if (stem == expected) {
      std::cout << file_path.string() << "\t" << "good" << std::endl;
    } else {
      std::cout << file_path.string() << "\t" << "bad, should be " << expected << std::endl;
    }
  }
  return 0;
}

