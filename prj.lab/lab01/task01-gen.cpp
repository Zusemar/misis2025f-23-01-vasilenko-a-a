// task01-gen: generate test images and task01.lst
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>

namespace fs = std::filesystem;

static cv::Mat makeSampleImage(int width, int height, int cvType) {
  int channels = CV_MAT_CN(cvType);
  cv::Mat img(height, width, cvType);
  // Fill with simple gradients per channel
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        double value = (double)(x + y + c * 10) / (double)(width + height) * 0.9;
        switch (CV_MAT_DEPTH(cvType)) {
          case CV_8U:  img.at<cv::Vec<unsigned char, 1>>(y, x)[0] = (unsigned char)(value * 255.0); break;
          case CV_8S:  img.at<cv::Vec<signed char, 1>>(y, x)[0] = (signed char)(value * 120.0 - 60.0); break;
          case CV_16U: img.at<cv::Vec<unsigned short, 1>>(y, x)[0] = (unsigned short)(value * 60000.0); break;
          case CV_16S: img.at<cv::Vec<short, 1>>(y, x)[0] = (short)(value * 20000.0 - 10000.0); break;
          case CV_32S: img.at<cv::Vec<int, 1>>(y, x)[0] = (int)(value * 100000.0 - 50000.0); break;
          case CV_32F: img.at<cv::Vec<float, 1>>(y, x)[0] = (float)(value); break;
          case CV_64F: img.at<cv::Vec<double, 1>>(y, x)[0] = (double)(value); break;
        }
      }
    }
  }
  // If multi-channel, convert from single channel to desired channels by merging
  if (channels > 1) {
    cv::Mat single(height, width, CV_MAKETYPE(CV_MAT_DEPTH(cvType), 1));
    // Fill single channel similarly
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        double value = (double)(x + y) / (double)(width + height) * 0.9;
        switch (CV_MAT_DEPTH(cvType)) {
          case CV_8U:  single.at<unsigned char>(y, x) = (unsigned char)(value * 255.0); break;
          case CV_8S:  single.at<signed char>(y, x) = (signed char)(value * 120.0 - 60.0); break;
          case CV_16U: single.at<unsigned short>(y, x) = (unsigned short)(value * 60000.0); break;
          case CV_16S: single.at<short>(y, x) = (short)(value * 20000.0 - 10000.0); break;
          case CV_32S: single.at<int>(y, x) = (int)(value * 100000.0 - 50000.0); break;
          case CV_32F: single.at<float>(y, x) = (float)(value); break;
          case CV_64F: single.at<double>(y, x) = (double)(value); break;
        }
      }
    }
    std::vector<cv::Mat> planes(channels, single);
    cv::merge(planes, img);
  }
  return img;
}

int main(int argc, char** argv) {
  // Output directory next to this source: prj.lab/lab01/testdata
  fs::path base = fs::path(argv[0]).parent_path();
  // When executed from bin.rel or bin, go up to project root bin folder then to prj.lab/lab01
  // Safer: derive relative to current working dir: create prj.lab/lab01/testdata if exists
  fs::path projRoot = fs::current_path();
  fs::path outDir = projRoot / "prj.lab" / "lab01" / "testdata";
  fs::create_directories(outDir);

  struct Spec { int w; int h; int type; };
  // Minimal set covering various depths and channels
  std::vector<Spec> specs = {
    { 8, 8, CV_MAKETYPE(CV_8U, 1) },
    { 16, 12, CV_MAKETYPE(CV_8U, 3) },
    { 10, 10, CV_MAKETYPE(CV_16U, 1) },
    { 14, 9, CV_MAKETYPE(CV_16S, 1) },
    { 12, 12, CV_MAKETYPE(CV_32S, 1) },
    { 12, 12, CV_MAKETYPE(CV_32F, 1) },
    { 12, 12, CV_MAKETYPE(CV_64F, 1) },
  };

  std::vector<std::string> savedNames;
  for (const auto& s : specs) {
    cv::Mat img = makeSampleImage(s.w, s.h, s.type);
    std::string baseName = strid_from_mat(img, 4);
    // Save in three formats: png, tiff, jpg
    std::vector<std::string> exts = { ".png", ".tiff", ".jpg" };
    for (const auto& ext : exts) {
      fs::path fn = outDir / (baseName + ext);
      cv::imwrite(fn.string(), img);
      savedNames.push_back(fn.filename().string());
    }
  }

  // Write task01.lst next to generated files
  fs::path lstPath = outDir / "task01.lst";
  std::ofstream lst(lstPath);
  for (const auto& name : savedNames) {
    lst << name << "\n";
  }
  lst.close();

  std::cout << "Generated " << savedNames.size() << " files in: " << outDir << std::endl;
  std::cout << "List written to: " << lstPath << std::endl;
  return 0;
}


