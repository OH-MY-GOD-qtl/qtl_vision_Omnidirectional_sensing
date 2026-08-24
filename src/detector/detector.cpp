#include "detector/detector.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <limits>

#include "tools/yaml.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "image/image.hpp"


Detector::Detector(const std::string & config_path, bool debug)
: classifier_(config_path), debug_(debug)
{
    auto yaml = yaml_load(config_path);

    // 分离颜色通道双阈值预处理参数（缺失键回退默认值，兼容旧配置）
    auto enemy_color =
        (yaml_read<std::string>(yaml, "enemy_color") == "red") ? Color::red : Color::blue;
    auto color_threshold = yaml["color_threshold"] ? yaml["color_threshold"].as<double>() : 40.0;
    auto brightness_threshold =
        yaml["brightness_threshold"] ? yaml["brightness_threshold"].as<double>() : 100.0;
    channel_thresholder_.configure(enemy_color, color_threshold, brightness_threshold);

    lightbar_params_.max_angle_error = yaml_read<double>(yaml, "max_angle_error") / 57.3;
    lightbar_params_.min_lightbar_ratio = yaml_read<double>(yaml, "min_lightbar_ratio");
    lightbar_params_.max_lightbar_ratio = yaml_read<double>(yaml, "max_lightbar_ratio");
    lightbar_params_.min_lightbar_length = yaml_read<double>(yaml, "min_lightbar_length");
    min_armor_ratio_ = yaml_read<double>(yaml, "min_armor_ratio");
    max_armor_ratio_ = yaml_read<double>(yaml, "max_armor_ratio");
    max_side_ratio_ = yaml_read<double>(yaml, "max_side_ratio");
    min_confidence_ = yaml_read<double>(yaml, "min_confidence");
    max_rectangular_error_ = yaml_read<double>(yaml, "max_rectangular_error") / 57.3;  // degree to rad

    save_path_ = "patterns";
    std::filesystem::create_directory(save_path_);
}

std::vector<Armor> Detector::detect(const cv::Mat & bgr_img, int frame_count)
{
    // 分离颜色通道双阈值二值化（图像预处理见 image 模块；缓冲帧间复用）
    channel_thresholder_.binary(bgr_img, binary_img_);
    if (debug_) cv::imshow("binary_img", binary_img_);

    // 获取轮廓点（容器复用）
    contours_.clear();
    cv::findContours(binary_img_, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 获取灯条（提取与几何筛选见 lightbar 模块）
    auto lightbars = extract_lightbars(bgr_img, contours_, lightbar_params_);

    // 节流诊断日志：轮廓/灯条数量（定位阈值与几何筛选问题）
    static int dbg_count = 0;
    if (++dbg_count % 30 == 0)
        logger()->debug("[Detector] contours={} lightbars={}", contours_.size(), lightbars.size());

    // 将灯条从左到右排序
    std::sort(
        lightbars.begin(), lightbars.end(),
        [](const Lightbar & a, const Lightbar & b) { return a.center.x < b.center.x; });

    // 获取装甲板
    std::vector<Armor> armors;
    for (auto left = lightbars.begin(); left != lightbars.end(); left++) {
        for (auto right = std::next(left); right != lightbars.end(); right++) {
            if (left->color != right->color) continue;

            auto armor = Armor(*left, *right);
            if (!check_geometry(armor)) {
                static int geom_dbg = 0;
                if (++geom_dbg % 30 == 0)
                    logger()->debug(
                        "[Detector] geom fail ratio={:.2f} side={:.2f} rect={:.1f}", armor.ratio,
                        armor.side_ratio, armor.rectangular_error * 57.3);
                continue;
            }

            armor.pattern = get_pattern(bgr_img, armor);
            classifier_.classify(armor);
            if (!check_name(armor, true)) {
                static int name_dbg = 0;
                if (++name_dbg % 5 == 0)
                    logger()->debug(
                        "[Detector] name fail: name={} conf={:.2f} ratio={:.2f}", ARMOR_NAMES[armor.name],
                        armor.confidence, armor.ratio);
                continue;
            }

            armor.type = get_type(armor);
            if (!check_type(armor)) continue;

            armor.center_norm = get_center_norm(bgr_img, armor.center);
            armors.emplace_back(armor);
        }
    }

    // 检查装甲板是否存在共用灯条的情况
    for (auto armor1 = armors.begin(); armor1 != armors.end(); armor1++) {
        for (auto armor2 = std::next(armor1); armor2 != armors.end(); armor2++) {
            if (
                armor1->left.id != armor2->left.id && armor1->left.id != armor2->right.id &&
                armor1->right.id != armor2->left.id && armor1->right.id != armor2->right.id) {
                continue;
            }

            // 装甲板重叠, 保留roi小的
            if (armor1->left.id == armor2->left.id || armor1->right.id == armor2->right.id) {
                auto area1 = armor1->pattern.cols * armor1->pattern.rows;
                auto area2 = armor2->pattern.cols * armor2->pattern.rows;
                if (area1 < area2)
                    armor2->duplicated = true;
                else
                    armor1->duplicated = true;
            }

            // 装甲板相连，保留置信度大的
            if (armor1->left.id == armor2->right.id || armor1->right.id == armor2->left.id) {
                if (armor1->confidence < armor2->confidence)
                    armor1->duplicated = true;
                else
                    armor2->duplicated = true;
            }
        }
    }

    armors.erase(
        std::remove_if(armors.begin(), armors.end(), [&](const Armor & a) { return a.duplicated; }),
        armors.end());

    if (debug_) show_result(binary_img_, bgr_img, lightbars, armors, frame_count);

    return armors;
}

bool Detector::detect(Armor & armor, const cv::Mat & bgr_img)
{
    // 取得四个角点
    auto tl = armor.points[0];
    auto tr = armor.points[1];
    auto br = armor.points[2];
    auto bl = armor.points[3];
    // 计算向量和调整后的点
    auto lt2b = bl - tl;
    auto rt2b = br - tr;
    auto tl1 = (tl + bl) / 2 - lt2b;
    auto bl1 = (tl + bl) / 2 + lt2b;
    auto br1 = (tr + br) / 2 + rt2b;
    auto tr1 = (tr + br) / 2 - rt2b;
    auto tl2tr = tr1 - tl1;
    auto bl2br = br1 - bl1;
    auto tl2 = (tl1 + tr) / 2 - 0.75 * tl2tr;
    auto tr2 = (tl1 + tr) / 2 + 0.75 * tl2tr;
    auto bl2 = (bl1 + br) / 2 - 0.75 * bl2br;
    auto br2 = (bl1 + br) / 2 + 0.75 * bl2br;
    // 构造新的四个角点
    std::vector<cv::Point> points = {tl2, tr2, br2, bl2};
    auto armor_rotaterect = cv::minAreaRect(points);
    cv::Rect boundingBox = armor_rotaterect.boundingRect();
    // 检查boundingBox是否超出图像边界
    if (
        boundingBox.x < 0 || boundingBox.y < 0 || boundingBox.x + boundingBox.width > bgr_img.cols ||
        boundingBox.y + boundingBox.height > bgr_img.rows) {
        return false;
    }

    // 在图像上裁剪出这个矩形区域（ROI）
    cv::Mat armor_roi = bgr_img(boundingBox);
    if (armor_roi.empty()) {
        return false;
    }

    // 分离颜色通道双阈值二值化 + 轮廓（图像预处理见 image 模块）
    auto binary_img = channel_thresholder_.binary(armor_roi);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 获取灯条（提取与几何筛选见 lightbar 模块）。
    // 颜色取样在 ROI 坐标系内进行（原实现把 ROI 局部轮廓坐标用于全图取色，存在坐标偏差）
    auto lightbars = extract_lightbars(armor_roi, contours, lightbar_params_);
    // correct_lightbar_points(lightbar, gray_img); //关闭PCA（gray_img 为 armor_roi 的灰度图）

    if (lightbars.size() < 2) return false;

    // 将灯条从左到右排序
    std::sort(
        lightbars.begin(), lightbars.end(),
        [](const Lightbar & a, const Lightbar & b) { return a.center.x < b.center.x; });

    // 计算与 tl_roi, bl_roi 和 br_roi, tr_roi 距离最近的灯条
    Lightbar * closest_left_lightbar = nullptr;
    Lightbar * closest_right_lightbar = nullptr;
    float min_distance_tl_bl = std::numeric_limits<float>::max();
    float min_distance_br_tr = std::numeric_limits<float>::max();
    for (auto & lightbar : lightbars) {
        float distance_tl_bl =
            cv::norm(tl - (lightbar.top + cv::Point2f(boundingBox.x, boundingBox.y))) +
            cv::norm(bl - (lightbar.bottom + cv::Point2f(boundingBox.x, boundingBox.y)));
        if (distance_tl_bl < min_distance_tl_bl) {
            min_distance_tl_bl = distance_tl_bl;
            closest_left_lightbar = &lightbar;
        }
        float distance_br_tr =
            cv::norm(br - (lightbar.bottom + cv::Point2f(boundingBox.x, boundingBox.y))) +
            cv::norm(tr - (lightbar.top + cv::Point2f(boundingBox.x, boundingBox.y)));
        if (distance_br_tr < min_distance_br_tr) {
            min_distance_br_tr = distance_br_tr;
            closest_right_lightbar = &lightbar;
        }
    }

    if (
        closest_left_lightbar && closest_right_lightbar &&
        min_distance_br_tr + min_distance_tl_bl < 15) {
        // 将四个点从armor_roi坐标系转换到原始图像坐标系
        armor.points[0] = closest_left_lightbar->top + cv::Point2f(boundingBox.x, boundingBox.y);
        armor.points[1] = closest_right_lightbar->top + cv::Point2f(boundingBox.x, boundingBox.y);
        armor.points[2] = closest_right_lightbar->bottom + cv::Point2f(boundingBox.x, boundingBox.y);
        armor.points[3] = closest_left_lightbar->bottom + cv::Point2f(boundingBox.x, boundingBox.y);
        return true;
    }

    return false;
}

bool Detector::check_geometry(const Armor & armor) const
{
    auto ratio_ok = armor.ratio > min_armor_ratio_ && armor.ratio < max_armor_ratio_;
    auto side_ratio_ok = armor.side_ratio < max_side_ratio_;
    auto rectangular_error_ok = armor.rectangular_error < max_rectangular_error_;
    return ratio_ok && side_ratio_ok && rectangular_error_ok;
}

bool Detector::check_name(const Armor & armor, bool save_uncertain) const
{
    auto name_ok = armor.name != ArmorName::not_armor;
    auto confidence_ok = armor.confidence > min_confidence_;

    // 保存不确定的图案，用于分类器的迭代
    if (save_uncertain && name_ok && !confidence_ok) save(armor);

    // 出现 5号 则显示 debug 信息。但不过滤。
    if (armor.name == ArmorName::five) logger()->debug("See pattern 5");

    return name_ok && confidence_ok;
}

bool Detector::check_type_neural(const Armor & armor)
{
    auto name_ok = (armor.type == ArmorType::small)
                                        ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                                        : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                                            armor.name != ArmorName::outpost);

    return name_ok;
}

ArmorType Detector::get_type_neural(const Armor & armor)
{
    // 英雄、基地只能是大装甲板
    if (armor.name == ArmorName::one || armor.name == ArmorName::base) {
        return ArmorType::big;
    }

    // 其余(工程、哨兵、前哨站、步兵)均为小装甲板
    return ArmorType::small;
}

cv::Mat Detector::get_pattern_neural(const cv::Mat & bgr_img, const Armor & armor)
{
    // 延长灯条获得装甲板角点
    // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
    auto tl = (armor.points[0] + armor.points[3]) / 2 - (armor.points[3] - armor.points[0]) * 1.125;
    auto bl = (armor.points[0] + armor.points[3]) / 2 + (armor.points[3] - armor.points[0]) * 1.125;
    auto tr = (armor.points[2] + armor.points[1]) / 2 - (armor.points[2] - armor.points[1]) * 1.125;
    auto br = (armor.points[2] + armor.points[1]) / 2 + (armor.points[2] - armor.points[1]) * 1.125;

    auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
    auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
    auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
    auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
    auto roi_tl = cv::Point(roi_left, roi_top);
    auto roi_br = cv::Point(roi_right, roi_bottom);
    auto roi = cv::Rect(roi_tl, roi_br);

    // 检查ROI是否有效
    if (roi_left < 0 || roi_top < 0 || roi_right <= roi_left || roi_bottom <= roi_top) {
        return cv::Mat();  // 返回一个空的Mat对象
    }

    // 检查ROI是否超出图像边界
    if (roi_right > bgr_img.cols || roi_bottom > bgr_img.rows) {
        return cv::Mat();  // 返回一个空的Mat对象
    }

    return bgr_img(roi);
}

bool Detector::check_type(const Armor & armor) const
{
    auto name_ok = armor.type == ArmorType::small
                                        ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                                        : (armor.name == ArmorName::one || armor.name == ArmorName::base);

    // 保存异常的图案，用于分类器的迭代
    if (!name_ok) {
        logger()->debug(
            "see strange armor: {} {}", ARMOR_TYPES[armor.type], ARMOR_NAMES[armor.name]);
        save(armor);
    }

    return name_ok;
}

cv::Mat Detector::get_pattern(const cv::Mat & bgr_img, const Armor & armor) const
{
    // 延长灯条获得装甲板角点
    // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
    auto tl = armor.left.center - armor.left.top2bottom * 1.125;
    auto bl = armor.left.center + armor.left.top2bottom * 1.125;
    auto tr = armor.right.center - armor.right.top2bottom * 1.125;
    auto br = armor.right.center + armor.right.top2bottom * 1.125;

    auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
    auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
    auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
    auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
    auto roi_tl = cv::Point(roi_left, roi_top);
    auto roi_br = cv::Point(roi_right, roi_bottom);
    auto roi = cv::Rect(roi_tl, roi_br);

    return bgr_img(roi);
}

ArmorType Detector::get_type(const Armor & armor)
{
    /// 优先根据当前armor.ratio判断
    /// TODO: 25赛季是否还需要根据比例判断大小装甲？能否根据图案直接判断？

    if (armor.ratio > 3.0) {
        // logger()->debug(
        //   "[Detector] get armor type by ratio: BIG {} {:.2f}", ARMOR_NAMES[armor.name], armor.ratio);
        return ArmorType::big;
    }

    if (armor.ratio < 2.5) {
        // logger()->debug(
        //   "[Detector] get armor type by ratio: SMALL {} {:.2f}", ARMOR_NAMES[armor.name], armor.ratio);
        return ArmorType::small;
    }

    // logger()->debug("[Detector] get armor type by name: {}", ARMOR_NAMES[armor.name]);

    // 英雄、基地只能是大装甲板
    if (armor.name == ArmorName::one || armor.name == ArmorName::base) {
        return ArmorType::big;
    }

    // 其他所有（工程、哨兵、前哨站、步兵）都是小装甲板
    /// TODO: 基地顶装甲是小装甲板
    return ArmorType::small;
}

cv::Point2f Detector::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
    auto h = bgr_img.rows;
    auto w = bgr_img.cols;
    return {center.x / w, center.y / h};
}

void Detector::classify(Armor & armor) { classifier_.classify(armor); }

void Detector::save(const Armor & armor) const
{
    auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
    auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, ARMOR_NAMES[armor.name], file_name);
    cv::imwrite(img_path, armor.pattern);
}

void Detector::show_result(
    const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::vector<Lightbar> & lightbars,
    const std::vector<Armor> & armors, int frame_count) const
{
    auto detection = bgr_img.clone();
    draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});

    for (const auto & lightbar : lightbars) {
        auto info = fmt::format(
            "{:.1f} {:.1f} {:.1f} {}", lightbar.angle_error * 57.3, lightbar.ratio, lightbar.length,
            COLORS[lightbar.color]);
        draw_text(detection, info, lightbar.top, {0, 255, 255});
        draw_points(detection, lightbar.points, {0, 255, 255}, 3);
    }

    for (const auto & armor : armors) {
        auto info = fmt::format(
            "{:.2f} {:.2f} {:.1f} {:.2f} {} {}", armor.ratio, armor.side_ratio,
            armor.rectangular_error * 57.3, armor.confidence, ARMOR_NAMES[armor.name],
            ARMOR_TYPES[armor.type]);
        draw_points(detection, armor.points, {0, 255, 0});
        draw_text(detection, info, armor.left.bottom, {0, 255, 0});
    }

    cv::Mat binary_img2;
    cv::resize(binary_img, binary_img2, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::resize(detection, detection, {}, 0.5, 0.5);     // 显示时缩小图片尺寸

    // cv::imshow("threshold", binary_img2);
    cv::imshow("detection", detection);
}

