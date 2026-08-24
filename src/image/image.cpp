#include "image/image.hpp"


cv::Mat to_gray(const cv::Mat & bgr)
{
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

void to_gray(const cv::Mat & bgr, cv::Mat & dst)
{
    cv::cvtColor(bgr, dst, cv::COLOR_BGR2GRAY);
}

cv::Mat to_binary(const cv::Mat & gray, double threshold)
{
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, 255, cv::THRESH_BINARY);
    return binary;
}

void to_binary(const cv::Mat & gray, double threshold, cv::Mat & dst)
{
    cv::threshold(gray, dst, threshold, 255, cv::THRESH_BINARY);
}

cv::Mat dilate_binary(const cv::Mat & binary, const cv::Size & kernel_size, int iterations)
{
    cv::Mat dilated;
    dilate_binary(binary, kernel_size, dilated, iterations);
    return dilated;
}

void dilate_binary(
    const cv::Mat & binary, const cv::Size & kernel_size, cv::Mat & dst, int iterations)
{
    auto kernel = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    cv::dilate(binary, dst, kernel, cv::Point(-1, -1), iterations);
}

cv::Rect letterbox(const cv::Mat & img, cv::Mat & dst, double & scale)
{
    // 等比缩放系数（与原实现一致：目标边长 / 图像行列取小）
    scale = std::min(
        static_cast<double>(dst.cols) / img.cols, static_cast<double>(dst.rows) / img.rows);
    auto w = static_cast<int>(img.cols * scale);
    auto h = static_cast<int>(img.rows * scale);

    // 退化输入（缩放后宽或高为 0）：不写 dst，返回空 roi
    if (w == 0 || h == 0) return {};

    // dst 由调用方零初始化；只写 roi 区域，roi 之外保持 0，无需逐帧 memset
    auto roi = cv::Rect(0, 0, w, h);
    cv::resize(img, dst(roi), {w, h});
    return roi;
}


ChannelThresholder::ChannelThresholder(
    Color enemy_color, double color_threshold, double brightness_threshold)
: enemy_color_(enemy_color),
    color_threshold_(color_threshold),
    brightness_threshold_(brightness_threshold)
{
    kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
}

void ChannelThresholder::configure(
    Color enemy_color, double color_threshold, double brightness_threshold)
{
    enemy_color_ = enemy_color;
    color_threshold_ = color_threshold;
    brightness_threshold_ = brightness_threshold;
}

cv::Mat ChannelThresholder::binary(const cv::Mat & bgr)
{
    cv::Mat out;
    binary(bgr, out);
    return out;
}

void ChannelThresholder::binary(const cv::Mat & bgr, cv::Mat & dst)
{
    // BGR 通道顺序：0=B 1=G 2=R；红敌取 R-B，蓝敌取 B-R（饱和差，负值截 0）
    cv::split(bgr, channels_);
    const int color_idx = (enemy_color_ == Color::red) ? 2 : 0;
    const int base_idx = (enemy_color_ == Color::red) ? 0 : 2;
    cv::subtract(channels_[color_idx], channels_[base_idx], diff_);

    // 颜色掩码：颜色差超过阈值
    cv::threshold(diff_, color_mask_, color_threshold_, 255, cv::THRESH_BINARY);

    // 亮度掩码：灰度超过阈值
    cv::cvtColor(bgr, gray_, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_, bright_mask_, brightness_threshold_, 255, cv::THRESH_BINARY);

    // 双条件取交 + 两次膨胀
    cv::bitwise_and(color_mask_, bright_mask_, dst);
    cv::dilate(dst, dst, kernel_);
    cv::dilate(dst, dst, kernel_);
}
