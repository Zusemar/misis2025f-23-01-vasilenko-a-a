#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>
#include <iostream>
#include <fstream>
#include <array>
#include <iomanip>

struct RegionMasks {
    cv::Mat background; // фон в пределах плитки, вне квадрата
    cv::Mat square;     // квадрат БЕЗ круга (однородная область)
    cv::Mat circle;     // круг
};

static RegionMasks make_masks_for_one_tile(int tile_x) {
    const int size = 256;
    const int square_size = 209;
    const int circle_r = 83;

    RegionMasks m;
    m.background = cv::Mat::zeros(size, size * 4, CV_8UC1);
    m.square     = cv::Mat::zeros(size, size * 4, CV_8UC1);
    m.circle     = cv::Mat::zeros(size, size * 4, CV_8UC1);

    const int offset_x = tile_x * size;
    const cv::Rect tileR(offset_x, 0, size, size);  // ROI текущей плитки
    const int square_x = offset_x + (size - square_size) / 2; // 23 + offset_x
    const int square_y = (size - square_size) / 2;            // 23
    const cv::Rect sqr(square_x, square_y, square_size, square_size);
    const cv::Point c(offset_x + size/2, size/2);

    // circle
    cv::circle(m.circle, c, circle_r, cv::Scalar(255), cv::FILLED);

    // square ring: сначала весь квадрат, потом вычитаем круг
    m.square(sqr).setTo(255);
    cv::circle(m.square, c, circle_r, cv::Scalar(0), cv::FILLED); // убрать круг

    // background только в пределах плитки и вне квадрата
    m.background(tileR).setTo(255);
    m.background(sqr).setTo(0);

    return m;
}


// анализ и запись статистики в CSV
void analyze_and_save_stats(const cv::Mat& base,
    const std::array<int,3>& sigmas,
    const std::string& csv_path) {
std::vector<cv::Mat> imgs{base};
for (int s : sigmas) imgs.push_back(add_noise_gau(base, s));

// уровни яркости (L0,L1,L2) из твоего задания
const std::array<cv::Vec3i,4> levels = {
cv::Vec3i(0, 127, 255),
cv::Vec3i(20, 127, 235),
cv::Vec3i(55, 127, 200),
cv::Vec3i(90, 127, 165)
};

std::ofstream fout(csv_path);
if (!fout.is_open()) {
std::cerr << "failed to open csv for writing: " << csv_path << "\n";
return;
}

fout << "tile,region,sigma,L_theory,mean_exp,diff_mean,var_theory,var_exp,diff_var,count,stddev,min,max\n";
const char* region_names[] = {"background", "square", "circle"};

for (int tile = 0; tile < 4; ++tile) {
RegionMasks m = make_masks_for_one_tile(tile);
for (int i = 0; i < imgs.size(); ++i) {
int sigma = (i == 0 ? 0 : sigmas[i - 1]);
const cv::Mat& img = imgs[i];

for (int r = 0; r < 3; ++r) {
const cv::Mat& mask = (r == 0) ? m.background :
              (r == 1) ? m.square : m.circle;

// теоретические значения
double L_theory = (r == 0) ? levels[tile][0]
         : (r == 1) ? levels[tile][1]
                    : levels[tile][2];
double var_theory = double(sigma) * sigma;

// практические
PixelDistributionStats st = calc_distribution_stats(img, mask);
double diff_mean = st.mean - L_theory;
double diff_var = st.variance - var_theory;

fout << (tile + 1) << "," << region_names[r] << "," << sigma << ","
<< std::fixed << std::setprecision(3)
<< L_theory << "," << st.mean << "," << diff_mean << ","
<< var_theory << "," << st.variance << "," << diff_var << ","
<< st.count << "," << st.stddev << ","
<< st.minimum << "," << st.maximum << "\n";
}
}
}

fout.close();
std::cout << "✔ saved stats with theory to " << csv_path << "\n";
}

static cv::Mat hstack4(const cv::Mat& a, const cv::Mat& b, const cv::Mat& c, const cv::Mat& d) {
    std::vector<cv::Mat> v{a,b,c,d};
    cv::Mat out;
    cv::hconcat(v, out);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: task02 <output_image_path> [--hist]\n";
        return 1;
    }
    const std::string outPath = argv[1];
    const bool writeHist = (argc >= 3 && std::string(argv[2]) == "--hist");

    // 1) четыре набора уровней
    const std::array<cv::Vec3i,4> levels = {
        cv::Vec3i(0, 127, 255),
        cv::Vec3i(20, 127, 235),
        cv::Vec3i(55, 127, 200),
        cv::Vec3i(90, 127, 165)
    };

    // 2) генерим и склеиваем по горизонтали
    std::vector<cv::Mat> samples;
    for (auto lv : levels)
        samples.push_back(gen_tgtimg00(lv[0], lv[1], lv[2]));
    cv::Mat base;
    cv::hconcat(samples, base); // 256x1024

    // 3) зашумлённые версии
    const std::array<int,3> sigmas = {3,7,15};
    std::vector<cv::Mat> stackRows { base };
    for (int s : sigmas)
        stackRows.push_back(add_noise_gau(base, s));

    // 4) склейка по вертикали
    cv::Mat finalImg;
    cv::vconcat(stackRows, finalImg); // 1024x1024

    // 5) сохраняем изображение
    if (!cv::imwrite(outPath, finalImg)) {
        std::cerr << "failed to save: " << outPath << "\n";
        return 2;
    }

    // 6) анализ статистик и CSV
    std::string csvPath = outPath.substr(0, outPath.find_last_of('.')) + "_stats.csv";
    analyze_and_save_stats(base, sigmas, csvPath);

    // 7) гистограммы (если нужно)
    if (writeHist) {
        auto histFor = [&](const cv::Mat& m, bool light) {
            return draw_histogram_8u(m,
                light ? cv::Scalar(235,235,235) : cv::Scalar(210,210,210),
                cv::Scalar(32,32,32));
        };

        cv::Mat histBase = histFor(base, true);
        cv::Mat hist3    = histFor(stackRows[1], false);
        cv::Mat hist7    = histFor(stackRows[2], true);
        cv::Mat hist15   = histFor(stackRows[3], false);

        auto tileRow = [&](const cv::Mat& h) {
            cv::Mat h1,h2,h3,h4,row;
            cv::resize(h, h1, cv::Size(256,256));
            cv::resize(h, h2, cv::Size(256,256));
            cv::resize(h, h3, cv::Size(256,256));
            cv::resize(h, h4, cv::Size(256,256));
            cv::hconcat(std::vector<cv::Mat>{h1,h2,h3,h4}, row);
            return row;
        };

        cv::Mat histMosaic;
        cv::vconcat(std::vector<cv::Mat>{
            tileRow(histBase), tileRow(hist3), tileRow(hist7), tileRow(hist15)
        }, histMosaic);

        std::string histPath = outPath;
        auto dot = histPath.find_last_of('.');
        if (dot == std::string::npos) histPath += "_hist.png";
        else histPath.insert(dot, "_hist");
        cv::imwrite(histPath, histMosaic);
    }

    std::cout << "✔ saved image to " << outPath << "\n";
    return 0;
}
