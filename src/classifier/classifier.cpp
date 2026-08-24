#include "classifier/classifier.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include "image/image.hpp"
#include "tools/yaml.hpp"


Classifier::Classifier(const std::string & config_path)
{
    auto yaml = yaml_load(config_path);
    auto model = yaml_read<std::string>(yaml, "classify_model");

    // 仅使用原生 OpenVINO 推理（原先同时加载了 cv::dnn 与 OpenVINO 两份模型）
    auto ovmodel = core_.read_model(model);
    compiled_model_ = core_.compile_model(
        ovmodel, "AUTO", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

    // 预分配输入缓冲并复用 InferRequest，避免每个目标都创建推理请求
    input_tensor_ = ov::Tensor(ov::element::f32, {1, 1, kInputSize, kInputSize});
    infer_request_ = compiled_model_.create_infer_request();
    infer_request_.set_input_tensor(input_tensor_);

    letterbox_u8_ = cv::Mat(kInputSize, kInputSize, CV_8UC1, cv::Scalar(0));
    input_f32_ = cv::Mat(kInputSize, kInputSize, CV_32FC1, input_tensor_.data());
    input_f32_.setTo(0.0f);
}

void Classifier::classify(Armor & armor)
{
    if (armor.pattern.empty()) {
        armor.name = ArmorName::not_armor;
        return;
    }

    to_gray(armor.pattern, gray_);

    // letterbox 等比缩放（预处理算法见 image 模块）；退化输入返回空 roi 且不写缓冲
    double scale;
    auto roi = letterbox(gray_, letterbox_u8_, scale);
    if (roi.empty()) {
        armor.name = ArmorName::not_armor;
        return;
    }

    // 分类器输入尺寸逐帧变化，letterbox roi 大小不固定：先清空输入缓冲，
    // 避免上一帧大图案残留污染本帧 padding 区域
    input_f32_.setTo(0.0f);
    letterbox_u8_(roi).convertTo(input_f32_(roi), CV_32F, 1.0 / 255.0);

    infer_request_.infer();

    // softmax + argmax（softmax 单调，argmax 可直接在原始分数上取，结果与原来一致）
    const auto & output_tensor = infer_request_.get_output_tensor();
    const float * scores = output_tensor.data<const float>();
    auto num_scores = output_tensor.get_size();

    std::size_t label_id = 0;
    for (std::size_t i = 1; i < num_scores; ++i) {
        if (scores[i] > scores[label_id]) label_id = i;
    }
    const float max_score = scores[label_id];
    double sum = 0.0;
    for (std::size_t i = 0; i < num_scores; ++i) sum += std::exp(scores[i] - max_score);

    armor.confidence = static_cast<float>(1.0 / sum);
    armor.name = static_cast<ArmorName>(label_id);
}

