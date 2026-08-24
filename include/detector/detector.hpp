#ifndef AUTO_AIM__DETECTOR_HPP
#define AUTO_AIM__DETECTOR_HPP

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "armor/armor.hpp"
#include "classifier/classifier.hpp"
#include "image/image.hpp"
#include "lightbar/lightbar.hpp"



class Detector
{
public:
    Detector(const std::string & config_path, bool debug = true, bool use_classifier = true);

    std::vector<Armor> detect(const cv::Mat & bgr_img, int frame_count = -1);

    // 用传统方法在装甲板附近 ROI 内二次提取灯条，矫正神经网络输出的四个角点
    bool detect(Armor & armor, const cv::Mat & bgr_img);

    // ---- 供神经网络检测器(YOLOV5/V8/V11)复用的公共判定 ----
    // 名称/置信度过滤; save_uncertain 控制是否保存低置信度图案(传统流程开启)
    bool check_name(const Armor & armor, bool save_uncertain) const;
    // 神经网络语义: 类型-名称一致性检查
    static bool check_type_neural(const Armor & armor);
    // 神经网络语义: 英雄/基地为大装甲板, 其余为小装甲板
    static ArmorType get_type_neural(const Armor & armor);
    // 由神经网络输出的四个角点裁剪装甲板图案
    static cv::Mat get_pattern_neural(const cv::Mat & bgr_img, const Armor & armor);
    cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;
    // 数字分类器（供 YOLO 链路复用，避免 YOLO 另建一份分类器导致模型重复加载）
    void classify(Armor & armor);

private:
    std::unique_ptr<Classifier> classifier_;  // 数字分类器（use_classifier=false 时为空）

    ChannelThresholder channel_thresholder_;  // 分离颜色通道双阈值预处理（见 image 模块）
    LightbarParams lightbar_params_;          // 灯条几何筛选参数（见 lightbar 模块）
    double min_armor_ratio_, max_armor_ratio_;
    double max_side_ratio_;
    double min_confidence_;
    double max_rectangular_error_;

    bool debug_;
    bool use_classifier_;  // 是否使用数字分类器（USB 全向路线为 false）
    std::string save_path_;

    // 帧级复用缓冲：避免每帧重复分配二值图与轮廓容器
    cv::Mat binary_img_;
    std::vector<std::vector<cv::Point>> contours_;

    bool check_geometry(const Armor & armor) const;
    bool check_type(const Armor & armor) const;  // 传统(比例)语义: 类型-名称一致性检查

    cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor) const;  // 传统: 由灯条延伸
    ArmorType get_type(const Armor & armor);                                  // 传统: 比例优先

    void save(const Armor & armor) const;
    void show_result(
        const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::vector<Lightbar> & lightbars,
        const std::vector<Armor> & armors, int frame_count) const;
};


#endif  // AUTO_AIM__DETECTOR_HPP
