#ifndef IMAGE__IMAGE_HPP
#define IMAGE__IMAGE_HPP

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "common/color.hpp"


// ---- 图像预处理：灰度 / 二值 / 形态学 / letterbox / 通道双阈值 ----
//
// 全部提供写入预分配缓冲的 out-param 重载：调用方持有缓冲即可在帧间复用，
// 避免每帧重复分配内存（cv::cvtColor/cv::threshold 内部对尺寸一致的 dst 复用内存）。

// BGR 转灰度
cv::Mat to_gray(const cv::Mat & bgr);
void to_gray(const cv::Mat & bgr, cv::Mat & dst);

// 固定阈值二值化（THRESH_BINARY）
cv::Mat to_binary(const cv::Mat & gray, double threshold);
void to_binary(const cv::Mat & gray, double threshold, cv::Mat & dst);

// 矩形核膨胀
cv::Mat dilate_binary(const cv::Mat & binary, const cv::Size & kernel_size, int iterations = 1);
void dilate_binary(
    const cv::Mat & binary, const cv::Size & kernel_size, cv::Mat & dst, int iterations = 1);

// letterbox：等比缩放 img 后写入 dst 左上角，返回实际放置区域 roi 与缩放系数 scale。
// 约定：dst 为目标尺寸的零初始化缓冲，函数只写 roi 区域、不再逐帧清零
// （roi 之外恒为 0，因为 resize 不会写出 roi）。退化输入（缩放后宽或高为 0）
// 返回空 roi 且不写 dst。
cv::Rect letterbox(const cv::Mat & img, cv::Mat & dst, double & scale);


// 分离颜色通道双阈值预处理（与桌面 qtl_vision 的 PreProcess::process 算法一致）：
//   1. 通道分离后取颜色差（红敌 R-B / 蓝敌 B-R，饱和差），过 color_threshold 得颜色掩码；
//   2. 灰度图过 brightness_threshold 得亮度掩码；
//   3. 两掩码按位与，再用 3x3 矩形核膨胀两次，得到二值图。
// 像素须同时满足"敌方颜色占优 + 足够亮"。
class ChannelThresholder
{
public:
    ChannelThresholder(
        Color enemy_color = Color::blue, double color_threshold = 40,
        double brightness_threshold = 100);

    void configure(Color enemy_color, double color_threshold, double brightness_threshold);

    cv::Mat binary(const cv::Mat & bgr);
    void binary(const cv::Mat & bgr, cv::Mat & dst);

private:
    Color enemy_color_;
    double color_threshold_, brightness_threshold_;

    // 帧间复用缓冲
    std::vector<cv::Mat> channels_;
    cv::Mat diff_, color_mask_, gray_, bright_mask_, kernel_;
};


#endif  // IMAGE__IMAGE_HPP
