#ifndef AUTO_AIM__YOLOV5_HPP
#define AUTO_AIM__YOLOV5_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <memory>
#include <string>
#include <vector>

#include "armor/armor.hpp"
#include "detector/detector.hpp"
#include "detector/tensorrt_infer.hpp"
#include "detector/yolo.hpp"


class YOLOV5 : public YOLOBase
{
public:
    YOLOV5(const std::string & config_path, bool debug);

    std::vector<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

    std::vector<Armor> postprocess(
        double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
    std::string device_, model_path_;
    bool debug_, use_roi_, use_traditional_, use_tensorrt_;

    const int class_num_ = 13;
    const float nms_threshold_ = 0.3;
    const float score_threshold_ = 0.7;

    // ---- OpenVINO 后端（机器人端 CPU 部署，默认）----
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;  // 复用以避免逐帧创建推理请求的开销

    // ---- TensorRT 后端（WSL 本机 NVIDIA 显卡）----
    std::unique_ptr<TensorRTInfer> trt_;

    cv::Mat input_;               // 预分配的 640x640 letterbox 输入（两后端共用）
    std::vector<float> blob_;     // 1x3x640x640 f32 NCHW（TensorRT 输入，帧间复用）

    cv::Rect roi_;
    cv::Point2f offset_;

    Detector detector_;  // 复用其 check_name/check_type_neural 等公共判定

    std::vector<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

    void draw_detections(const cv::Mat & img, const std::vector<Armor> & armors, int frame_count) const;
    double sigmoid(double x);
};


#endif  //AUTO_AIM__YOLOV5_HPP