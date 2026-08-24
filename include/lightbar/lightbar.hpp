#ifndef LIGHTBAR__LIGHTBAR_HPP
#define LIGHTBAR__LIGHTBAR_HPP

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <vector>
#include <vector>

#include "common/color.hpp"


// 灯条：由旋转矩形拟合出的装甲板发光灯条模型（几何量由构造函数计算）
struct Lightbar
{
    std::size_t id;
    Color color;
    cv::Point2f center, top, bottom, top2bottom;
    std::vector<cv::Point2f> points;
    double angle, angle_error, length, width, ratio;
    cv::RotatedRect rotated_rect;

    Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id);
    Lightbar() {};
};

// 灯条几何筛选参数（来自 yaml：max_angle_error / min,max_lightbar_ratio / min_lightbar_length）
struct LightbarParams
{
    double max_angle_error = 45 / 57.3;  // rad
    double min_lightbar_ratio = 1.5;
    double max_lightbar_ratio = 20;
    double min_lightbar_length = 8;
};

// 从轮廓中提取灯条：最小外接矩形拟合 → 几何筛选 → 颜色判定
std::vector<Lightbar> extract_lightbars(
    const cv::Mat & bgr_img, const std::vector<std::vector<cv::Point>> & contours,
    const LightbarParams & params);

// 灯条几何筛选：角度误差、长宽比、最小长度
bool check_lightbar_geometry(const Lightbar & lightbar, const LightbarParams & params);

// 按轮廓像素的红蓝分量和判定灯条颜色
Color get_lightbar_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour);

// 利用 PCA 回归灯条角点，参考自 https://github.com/CSU-FYT-Vision/FYT2024_vision
void correct_lightbar_points(Lightbar & lightbar, const cv::Mat & gray_img);


#endif  // LIGHTBAR__LIGHTBAR_HPP
