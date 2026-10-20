#include "segmentation.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool ok,const char* what) {if(!ok)throw std::runtime_error(what);}
int main() {
    try {
        walnut::Parameters p;
        const cv::Mat uniform(71,83,CV_32F,cv::Scalar(0.6));
        auto filtered=walnut::guidedSelfFilter(uniform,4,0.001);
        require(cv::norm(uniform,filtered,cv::NORM_INF)<1e-5,"Guided filter must preserve a constant field");
        cv::Mat step=cv::Mat::zeros(80,80,CV_32F); step.colRange(40,80).setTo(1);
        filtered=walnut::guidedSelfFilter(step,4,0.001);
        require(filtered.at<float>(40,39)<0.03&&filtered.at<float>(40,40)>0.97,"Guided filter blurred a strong edge");
        for(const std::string kind:{"none","gaussian","guided"}) {
            p.filter=kind;
            auto blank=walnut::segment(cv::Mat(50,70,CV_32F,cv::Scalar(0.002)),p);
            require(blank.empty&&cv::countNonZero(blank.labels)==0,"Empty image must remain background");
        }
        cv::Mat synthetic(160,160,CV_32F,cv::Scalar(0));
        cv::circle(synthetic,{80,80},60,cv::Scalar(0.061),8);
        cv::circle(synthetic,{65,72},16,cv::Scalar(0.036),-1);
        cv::line(synthetic,{80,40},{80,120},cv::Scalar(0.055),2);
        for(const std::string kind:{"none","gaussian","guided"}) {
            p.filter=kind;
            auto result=walnut::segment(synthetic,p);
            require(result.labels.type()==CV_8U&&result.labels.size()==synthetic.size(),"Wrong labelmap geometry/type");
            require(result.labels.at<unsigned char>(5,5)==0,"Air class incorrect");
            require(result.labels.at<unsigned char>(20,80)==1,"Shell class incorrect");
            require(result.labels.at<unsigned char>(72,65)==2,"Kernel class incorrect");
            require(result.labels.at<unsigned char>(110,80)==1,"Thin septum class incorrect");
        }
        bool rejected=false;
        try {walnut::segment(cv::Mat(5,5,CV_8U,cv::Scalar(0)),p);} catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"Must reject uint8 input rather than silently rescale it");
        cv::Mat bad(5,5,CV_32F,cv::Scalar(0));bad.at<float>(1,1)=std::numeric_limits<float>::quiet_NaN();
        rejected=false;
        try{walnut::segment(bad,p);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Must reject NaN input");
        p.epsilon=0; rejected=false;
        try{walnut::validateParameters(p);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Must reject invalid parameters");
        std::cout<<"Constant, edge preservation, empty images, synthetic class semantics and invalid inputs passed.\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
