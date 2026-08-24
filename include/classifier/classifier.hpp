#ifndef AUTO_AIM__CLASSIFIER_HPP
#define AUTO_AIM__CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "armor/armor.hpp"


class Classifier
{
public:
    explicit Classifier(const std::string & config_path);

    void classify(Armor & armor);

private:
    static constexpr int kInputSize = 32;  // 分类网络输入为 32x32 灰度图

    ov::Core core_;
    ov::CompiledModel compiled_model_;
    // 复用同一个 InferRequest 与输入缓冲，避免逐目标创建推理请求的开销
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;
    cv::Mat letterbox_u8_;  // 32x32 零填充灰度图
    cv::Mat input_f32_;     // 包装 input_tensor_ 内存的 f32 输入
    cv::Mat gray_;          // 灰度中间缓冲：多个目标连续分类时复用
};


#endif  // AUTO_AIM__CLASSIFIER_HPP
