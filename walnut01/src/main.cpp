#include "segmentation.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs=std::filesystem;
int main(int argc,char** argv) {
    try {
        cv::setNumThreads(1);
        if(argc<3) {
            std::cerr<<"Usage: walnut_segment <input.tif|directory> <output-directory> [--config file.json] [--filter none|gaussian|guided] [--method threshold|regions]\n";
            return 2;
        }
        std::string config,filter,method;
        for(int i=3;i<argc;i+=2) {
            if(i+1>=argc) throw std::invalid_argument("Missing option value");
            const std::string key=argv[i];
            if(key=="--config") config=argv[i+1];
            else if(key=="--filter") filter=argv[i+1];
            else if(key=="--method") method=argv[i+1];
            else throw std::invalid_argument("Unknown option: "+key);
        }
        auto params=walnut::loadParameters(config);
        if(!filter.empty()) params.filter=filter;
        if(!method.empty()) params.method=method;
        walnut::validateParameters(params);
        const fs::path source=argv[1],output=argv[2];
        std::vector<fs::path> files;
        if(fs::is_directory(source)) {
            for(const auto& entry:fs::directory_iterator(source)) {
                auto ext=entry.path().extension().string();
                std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return std::tolower(c);});
                if(entry.is_regular_file()&&(ext==".tif"||ext==".tiff")) files.push_back(entry.path());
            }
            std::sort(files.begin(),files.end());
        } else if(fs::is_regular_file(source)) files.push_back(source);
        if(files.empty()) throw std::runtime_error("No input TIFF files");
        fs::create_directories(output/"masks");
        fs::create_directories(output/"filtered");
        fs::create_directories(output/"intermediates");
        std::ofstream records(output/"diagnostics.csv"),regions(output/"regions.csv");
        if(!records||!regions) throw std::runtime_error("Cannot write output diagnostics");
        records<<"image,air_threshold,shell_threshold,empty,milliseconds\n";
        regions<<"image,area,mean,median,texture,label\n";
        records<<std::setprecision(10); regions<<std::setprecision(10);
        for(const auto& file:files) {
            const auto img=cv::imread(file.string(),cv::IMREAD_UNCHANGED);
            const auto result=walnut::segment(img,params);
            if(!cv::imwrite((output/"masks"/(file.stem().string()+".png")).string(),result.labels)||
               !cv::imwrite((output/"filtered"/(file.stem().string()+".tif")).string(),result.filtered))
                throw std::runtime_error("Could not write prediction: "+file.string());
            for(const auto& part:std::vector<std::pair<std::string,cv::Mat>>{
                {"material",result.material},{"cores",result.cores},{"markers",result.markers}}) {
                if(!cv::imwrite((output/"intermediates"/(file.stem().string()+"_"+part.first+".png")).string(),part.second))
                    throw std::runtime_error("Could not write intermediate: "+file.string());
            }
            records<<file.stem().string()<<','<<result.air_threshold<<','<<result.shell_threshold<<','<<result.empty<<','<<result.elapsed_ms<<'\n';
            for(const auto& r:result.regions) regions<<file.stem().string()<<','<<r.area<<','<<r.mean<<','<<r.median<<','<<r.texture<<','<<r.label<<'\n';
            std::cout<<file.filename().string()<<": "<<result.elapsed_ms<<" ms\n";
        }
        cv::FileStorage resolved((output/"parameters.json").string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
        if(!resolved.isOpened()) throw std::runtime_error("Cannot save resolved parameters");
        resolved<<"filter"<<params.filter<<"method"<<params.method<<"intensity_scale"<<params.intensity_scale
            <<"radius"<<params.radius<<"epsilon"<<params.epsilon<<"sigma"<<params.sigma
            <<"air_factor"<<params.air_factor<<"minimum_material"<<params.minimum_material<<"empty_peak"<<params.empty_peak
            <<"core_radius"<<params.core_radius<<"min_core_area"<<params.min_core_area
            <<"min_kernel_level"<<params.min_kernel_level<<"max_texture"<<params.max_texture<<"edge_weight"<<params.edge_weight
            <<"opencv_version"<<CV_VERSION<<"opencv_threads"<<cv::getNumThreads();
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"Error: "<<error.what()<<'\n';
        return 1;
    }
}
