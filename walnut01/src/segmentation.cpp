#include "segmentation.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace walnut {
namespace {
template<class T> void read(const cv::FileStorage& fs, const char* key, T& v) {
    if (!fs[key].empty()) fs[key] >> v;
}
cv::Mat element(int radius) {
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, {2*radius+1,2*radius+1});
}
cv::Mat filter(const cv::Mat& image, const Parameters& p) {
    if(p.filter=="none") return image.clone();
    if(p.filter=="guided") return guidedSelfFilter(image,p.radius,p.epsilon);
    cv::Mat out;
    cv::GaussianBlur(image,out,{0,0},p.sigma,p.sigma,cv::BORDER_REFLECT_101);
    return out;
}
cv::Mat materialMask(const cv::Mat& image, double threshold) {
    cv::Mat raw, cc, stats, centers;
    cv::compare(image,threshold,raw,cv::CMP_GT);
    const int count=cv::connectedComponentsWithStats(raw,cc,stats,centers,8,CV_32S);
    cv::Mat clean=cv::Mat::zeros(raw.size(),CV_8U);
    for(int i=1;i<count;++i) {
        if(stats.at<int>(i,cv::CC_STAT_AREA)>=3) clean.setTo(255,cc==i);
    }
    return clean;
}
cv::Mat regionLabels(const cv::Mat& filtered, const cv::Mat& original,
                     const cv::Mat& material, double shell_level,
                     const Parameters& p, std::vector<RegionInfo>& info,
                     cv::Mat& cores, cv::Mat& markers) {
    // Seeds are inferred from the input image alone. No annotation coordinates,
    // image identifiers, labelmaps or hand-painted envelopes enter this code.
    cv::Mat core, cc, stats, centers;
    cv::erode(material,core,element(p.core_radius));
    cores=core.clone();
    const int count=cv::connectedComponentsWithStats(core,cc,stats,centers,8,CV_32S);
    std::vector<std::vector<float>> samples(count);
    std::vector<double> texture_sum(count,0);
    cv::Mat mean, mean_sq, var;
    cv::boxFilter(original,mean,CV_32F,{7,7},{-1,-1},true,cv::BORDER_REFLECT_101);
    cv::boxFilter(original.mul(original),mean_sq,CV_32F,{7,7},{-1,-1},true,cv::BORDER_REFLECT_101);
    var=mean_sq-mean.mul(mean);
    cv::max(var,0,var);
    cv::sqrt(var,var);
    for(int y=0;y<cc.rows;++y) for(int x=0;x<cc.cols;++x) {
        const int id=cc.at<int>(y,x);
        if(id>0) {
            samples[id].push_back(filtered.at<float>(y,x));
            texture_sum[id]+=var.at<float>(y,x);
        }
    }
    std::vector<unsigned char> category(count,0);
    for(int i=1;i<count;++i) {
        const int area=stats.at<int>(i,cv::CC_STAT_AREA);
        if(area<p.min_core_area) continue;
        auto& values=samples[i];
        double sum=0; for(float v:values) sum+=v;
        std::nth_element(values.begin(),values.begin()+values.size()/2,values.end());
        const double median=values[values.size()/2];
        const double texture=texture_sum[i]/area;
        const bool kernel=median<shell_level && median>=p.min_kernel_level && texture<=p.max_texture;
        category[i]=kernel?2:1;
        info.push_back({area,sum/area,median,texture,category[i]});
    }
    // Multi-source shortest-path propagation within material. Crossing strong
    // intensity changes costs more than moving through a homogeneous region.
    const int width=filtered.cols, total=width*filtered.rows;
    std::vector<float> distance(total,std::numeric_limits<float>::infinity());
    cv::Mat labels=cv::Mat::zeros(filtered.size(),CV_8U);
    cv::Mat thickness, distance_to_core;
    cv::distanceTransform(material,thickness,cv::DIST_L2,cv::DIST_MASK_PRECISE);
    cv::distanceTransform(255-core,distance_to_core,cv::DIST_L2,cv::DIST_MASK_PRECISE);
    using Entry=std::pair<float,int>;
    std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> queue;
    for(int y=0;y<cc.rows;++y) for(int x=0;x<cc.cols;++x) {
        auto cls=category[cc.at<int>(y,x)];
        // A narrow septum can touch kernel without belonging to it. Seed its
        // distant thin stretches as shell so kernel cannot flood the whole wire.
        if(!cls && material.at<unsigned char>(y,x) &&
           thickness.at<float>(y,x)<=p.core_radius+0.5 &&
           distance_to_core.at<float>(y,x)>p.core_radius+2.0) cls=1;
        if(cls) {
            const int pos=y*width+x;
            labels.at<unsigned char>(y,x)=cls;
            distance[pos]=0;
            queue.push({0,pos});
        }
    }
    markers=labels.clone();
    const std::array<int,8> dx={-1,0,1,-1,1,-1,0,1};
    const std::array<int,8> dy={-1,-1,-1,0,0,1,1,1};
    while(!queue.empty()) {
        const auto [cost,pos]=queue.top(); queue.pop();
        if(cost>distance[pos]) continue;
        const int x=pos%width,y=pos/width;
        for(int k=0;k<8;++k) {
            const int xx=x+dx[k], yy=y+dy[k];
            if(xx<0 || yy<0 || xx>=width || yy>=filtered.rows || !material.at<unsigned char>(yy,xx)) continue;
            const int next=yy*width+xx;
            const float length=(dx[k]&&dy[k])?1.41421356f:1.f;
            const float delta=std::abs(filtered.at<float>(y,x)-filtered.at<float>(yy,xx));
            const float candidate=cost+length*(1.f+static_cast<float>(p.edge_weight)*delta);
            if(candidate<distance[next]) {
                distance[next]=candidate;
                labels.at<unsigned char>(yy,xx)=labels.at<unsigned char>(y,x);
                queue.push({candidate,next});
            }
        }
    }
    // Thin isolated material with no surviving core is treated as a septum.
    labels.setTo(1,(material!=0)&(labels==0));
    return labels;
}
} // namespace

void validateParameters(const Parameters& p) {
    auto finitePositive=[](double v){return std::isfinite(v)&&v>0;};
    if(p.filter!="none"&&p.filter!="guided"&&p.filter!="gaussian") throw std::invalid_argument("Unknown filter");
    if(p.method!="threshold"&&p.method!="regions") throw std::invalid_argument("Unknown method");
    if(!finitePositive(p.intensity_scale)||!finitePositive(p.epsilon)||!finitePositive(p.sigma)||
       !finitePositive(p.air_factor)||!finitePositive(p.minimum_material)||!finitePositive(p.empty_peak)||
       !finitePositive(p.min_kernel_level)||!finitePositive(p.max_texture)||
       !std::isfinite(p.edge_weight)||p.edge_weight<0||p.radius<1||p.radius>100||
       p.core_radius<1||p.core_radius>100||p.min_core_area<1)
        throw std::invalid_argument("Invalid segmentation parameters");
}
Parameters loadParameters(const std::string& file) {
    Parameters p;
    if(!file.empty()) {
        cv::FileStorage fs(file,cv::FileStorage::READ);
        if(!fs.isOpened()) throw std::runtime_error("Cannot open config: "+file);
        read(fs,"filter",p.filter); read(fs,"method",p.method);
        read(fs,"intensity_scale",p.intensity_scale); read(fs,"radius",p.radius);
        read(fs,"epsilon",p.epsilon); read(fs,"sigma",p.sigma);
        read(fs,"air_factor",p.air_factor); read(fs,"minimum_material",p.minimum_material);
        read(fs,"empty_peak",p.empty_peak); read(fs,"core_radius",p.core_radius);
        read(fs,"min_core_area",p.min_core_area); read(fs,"min_kernel_level",p.min_kernel_level);
        read(fs,"max_texture",p.max_texture); read(fs,"edge_weight",p.edge_weight);
    }
    validateParameters(p);
    return p;
}

cv::Mat guidedSelfFilter(const cv::Mat& input, int radius, double epsilon) {
    if(input.type()!=CV_32FC1||radius<1||!std::isfinite(epsilon)||epsilon<=0)
        throw std::invalid_argument("Guided filter requires float32 grayscale, radius>=1, epsilon>0");
    const cv::Size window(2*radius+1,2*radius+1);
    cv::Mat mean, corr, variance;
    cv::boxFilter(input,mean,CV_32F,window,{-1,-1},true,cv::BORDER_REFLECT_101);
    cv::boxFilter(input.mul(input),corr,CV_32F,window,{-1,-1},true,cv::BORDER_REFLECT_101);
    variance=corr-mean.mul(mean);
    cv::max(variance,0,variance);
    cv::Mat a,b,mean_a,mean_b;
    cv::divide(variance,variance+epsilon,a);
    b=mean-a.mul(mean);
    cv::boxFilter(a,mean_a,CV_32F,window,{-1,-1},true,cv::BORDER_REFLECT_101);
    cv::boxFilter(b,mean_b,CV_32F,window,{-1,-1},true,cv::BORDER_REFLECT_101);
    return mean_a.mul(input)+mean_b;
}

std::pair<double,double> multiOtsu(const cv::Mat& normalized) {
    if(normalized.type()!=CV_32FC1||normalized.empty()) throw std::invalid_argument("Invalid histogram input");
    std::array<double,256> hist{}, weight{}, moment{};
    for(int y=0;y<normalized.rows;++y) for(int x=0;x<normalized.cols;++x) {
        const int bin=std::clamp(static_cast<int>(normalized.at<float>(y,x)*255),0,255);
        hist[bin]++;
    }
    for(int i=0;i<256;++i) {
        weight[i]=hist[i]+(i?weight[i-1]:0);
        moment[i]=hist[i]*i+(i?moment[i-1]:0);
    }
    double best=-1; int t0=64,t1=180;
    for(int a=0;a<254;++a) for(int b=a+1;b<255;++b) {
        const double w0=weight[a],w1=weight[b]-weight[a],w2=weight[255]-weight[b];
        if(w0==0||w1==0||w2==0) continue;
        const double m0=moment[a],m1=moment[b]-moment[a],m2=moment[255]-moment[b];
        const double score=m0*m0/w0+m1*m1/w1+m2*m2/w2;
        if(score>best) {best=score;t0=a;t1=b;}
    }
    return {(t0+1)/255.0,(t1+1)/255.0};
}

Result segment(const cv::Mat& image, const Parameters& p) {
    validateParameters(p);
    if(image.empty()||image.type()!=CV_32FC1||!cv::checkRange(image,true,nullptr,0,1e6))
        throw std::invalid_argument("Input must be finite nonnegative single-channel float32 CT data");
    const auto begin=std::chrono::steady_clock::now();
    Result out;
    out.labels=cv::Mat::zeros(image.size(),CV_8U);
    out.material=cv::Mat::zeros(image.size(),CV_8U);
    out.cores=cv::Mat::zeros(image.size(),CV_8U);
    out.markers=cv::Mat::zeros(image.size(),CV_8U);
    const cv::Mat normalized=image/p.intensity_scale;
    const cv::Mat smoothed=filter(normalized,p);
    out.filtered=smoothed*p.intensity_scale;
    double peak=0; cv::minMaxLoc(image,nullptr,&peak);
    if(peak<p.empty_peak) {
        out.empty=true;
    } else {
        const auto [air,shell]=multiOtsu(smoothed);
        const double low=std::max(air*p.air_factor,p.minimum_material/p.intensity_scale);
        const double high=std::clamp(shell,0.042/p.intensity_scale,0.053/p.intensity_scale);
        out.air_threshold=low*p.intensity_scale;
        out.shell_threshold=high*p.intensity_scale;
        const cv::Mat material=materialMask(smoothed,low);
        out.material=material;
        if(p.method=="threshold") {
            out.labels.setTo(2,material);
            out.labels.setTo(1,(smoothed>=high)&material);
        } else {
            out.labels=regionLabels(smoothed,normalized,material,high,p,out.regions,out.cores,out.markers);
        }
    }
    out.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    return out;
}
} // namespace walnut
