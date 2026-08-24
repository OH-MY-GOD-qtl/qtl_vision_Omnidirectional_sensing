#ifndef AUTO_AIM__YOLOV8_HPP
#define AUTO_AIM__YOLOV8_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "armor/armor.hpp"
#include "detector/detector.hpp"
#include "detector/yolo.hpp"



class YOLOV8 : public YOLOBase
{
public:
    YOLOV8(const std::string & config_path, bool debug);

    std::vector<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

    std::vector<Armor> postprocess(
        double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
    // 数字分类器复用 detector_ 内部的实例（Detector::classify），避免 tiny_resnet 模型重复加载
    Detector detector_;

    std::string device_, model_path_;
    bool debug_, use_roi_;

    const int class_num_ = 2;
    const float nms_threshold_ = 0.3;
    const float score_threshold_ = 0.7;

    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;  // 复用以避免逐帧创建推理请求的开销
    cv::Mat input_;                   // 预分配的 416x416 letterbox 输入

    cv::Rect roi_;
    cv::Point2f offset_;

    std::vector<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

    void draw_detections(const cv::Mat & img, const std::vector<Armor> & armors, int frame_count) const;
    void sort_keypoints(std::vector<cv::Point2f> & keypoints);
};


#endif  // TOOLS__YOLOV8_HPP