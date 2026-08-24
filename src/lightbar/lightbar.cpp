#include "lightbar/lightbar.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>


Lightbar::Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id)
: id(id), rotated_rect(rotated_rect)
{
    std::vector<cv::Point2f> corners(4);
    rotated_rect.points(&corners[0]);
    std::sort(corners.begin(), corners.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
        return a.y < b.y;
    });

    center = rotated_rect.center;
    top = (corners[0] + corners[1]) / 2;
    bottom = (corners[2] + corners[3]) / 2;
    top2bottom = bottom - top;

    points.emplace_back(top);
    points.emplace_back(bottom);

    width = cv::norm(corners[0] - corners[1]);
    angle = std::atan2(top2bottom.y, top2bottom.x);
    angle_error = std::abs(angle - CV_PI / 2);
    length = cv::norm(top2bottom);
    ratio = length / width;
}

std::vector<Lightbar> extract_lightbars(
    const cv::Mat & bgr_img, const std::vector<std::vector<cv::Point>> & contours,
    const LightbarParams & params)
{
    std::size_t lightbar_id = 0;
    std::vector<Lightbar> lightbars;
    for (const auto & contour : contours) {
        auto lightbar = Lightbar(cv::minAreaRect(contour), lightbar_id);

        if (!check_lightbar_geometry(lightbar, params)) continue;

        lightbar.color = get_lightbar_color(bgr_img, contour);
        lightbars.emplace_back(lightbar);
        lightbar_id += 1;
    }
    return lightbars;
}

bool check_lightbar_geometry(const Lightbar & lightbar, const LightbarParams & params)
{
    auto angle_ok = lightbar.angle_error < params.max_angle_error;
    auto ratio_ok =
        lightbar.ratio > params.min_lightbar_ratio && lightbar.ratio < params.max_lightbar_ratio;
    auto length_ok = lightbar.length > params.min_lightbar_length;
    return angle_ok && ratio_ok && length_ok;
}

Color get_lightbar_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour)
{
    int red_sum = 0, blue_sum = 0;

    // 直接按行指针访问，避免每个点两次 at<> 的偏移计算开销（统计结果与原实现一致）
    for (const auto & point : contour) {
        const auto * px = bgr_img.ptr<cv::Vec3b>(point.y) + point.x;
        red_sum += (*px)[2];
        blue_sum += (*px)[0];
    }

    return blue_sum > red_sum ? Color::blue : Color::red;
}

void correct_lightbar_points(Lightbar & lightbar, const cv::Mat & gray_img)
{
    // 配置参数
    constexpr float MAX_BRIGHTNESS = 25;  // 归一化最大亮度值
    constexpr float ROI_SCALE = 0.07;     // ROI扩展比例
    constexpr float SEARCH_START = 0.4;   // 搜索起始位置比例（原0.8/2）
    constexpr float SEARCH_END = 0.6;     // 搜索结束位置比例（原1.2/2）

    // 扩展并裁剪ROI
    cv::Rect roi_box = lightbar.rotated_rect.boundingRect();
    roi_box.x -= roi_box.width * ROI_SCALE;
    roi_box.y -= roi_box.height * ROI_SCALE;
    roi_box.width += 2 * roi_box.width * ROI_SCALE;
    roi_box.height += 2 * roi_box.height * ROI_SCALE;

    // 边界约束
    roi_box &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);

    // 归一化ROI
    cv::Mat roi = gray_img(roi_box);
    const float mean_val = cv::mean(roi)[0];
    roi.convertTo(roi, CV_32F);
    cv::normalize(roi, roi, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

    // 计算质心
    const cv::Moments moments = cv::moments(roi);
    const cv::Point2f centroid(
        moments.m10 / moments.m00 + roi_box.x, moments.m01 / moments.m00 + roi_box.y);

    // 生成稀疏点云（优化性能）
    std::vector<cv::Point2f> points;
    for (int i = 0; i < roi.rows; ++i) {
        for (int j = 0; j < roi.cols; ++j) {
            const float weight = roi.at<float>(i, j);
            if (weight > 1e-3) {          // 忽略极小值提升性能
                points.emplace_back(j, i);  // 坐标相对于ROI区域
            }
        }
    }

    // PCA计算对称轴方向
    cv::PCA pca(cv::Mat(points).reshape(1), cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
    axis /= cv::norm(axis);
    if (axis.y > 0) axis = -axis;  // 统一方向

    const auto find_corner = [&](int direction) -> cv::Point2f {
        const float dx = axis.x * direction;
        const float dy = axis.y * direction;
        const float search_length = lightbar.length * (SEARCH_END - SEARCH_START);

        std::vector<cv::Point2f> candidates;

        // 横向采样多个候选线
        const int half_width = (lightbar.width - 2) / 2;
        for (int i_offset = -half_width; i_offset <= half_width; ++i_offset) {
            // 计算搜索起点
            cv::Point2f start_point(
                centroid.x + lightbar.length * SEARCH_START * dx + i_offset,
                centroid.y + lightbar.length * SEARCH_START * dy);

            // 沿轴搜索亮度跳变点
            cv::Point2f corner = start_point;
            float max_diff = 0;
            bool found = false;

            for (float step = 0; step < search_length; ++step) {
                const cv::Point2f cur_point(start_point.x + dx * step, start_point.y + dy * step);

                // 边界检查
                if (
                    cur_point.x < 0 || cur_point.x >= gray_img.cols || cur_point.y < 0 ||
                    cur_point.y >= gray_img.rows) {
                    break;
                }

                // 计算亮度差（使用双线性插值提升精度）
                const auto prev_val = gray_img.at<uchar>(cv::Point2i(cur_point - cv::Point2f(dx, dy)));
                const auto cur_val = gray_img.at<uchar>(cv::Point2i(cur_point));
                const float diff = prev_val - cur_val;

                if (diff > max_diff && prev_val > mean_val) {
                    max_diff = diff;
                    corner = cur_point - cv::Point2f(dx, dy);  // 跳变发生在上一位置
                    found = true;
                }
            }

            if (found) {
                candidates.push_back(corner);
            }
        }

        // 返回候选点均值
        return candidates.empty()
                            ? cv::Point2f(-1, -1)
                            : std::accumulate(candidates.begin(), candidates.end(), cv::Point2f(0, 0)) /
                                    static_cast<float>(candidates.size());
    };

    // 并行检测顶部和底部
    lightbar.top = find_corner(1);
    lightbar.bottom = find_corner(-1);
}
