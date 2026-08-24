#include "detector/yolos/yolov8.hpp"

#include <fmt/chrono.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <random>

#include "classifier/classifier.hpp"
#include "image/image.hpp"
#include "tools/yaml.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"


YOLOV8::YOLOV8(const std::string & config_path, bool debug)
: detector_(config_path), debug_(debug)
{
    auto yaml = yaml_load(config_path);

    model_path_ = yaml_read<std::string>(yaml, "yolov8_model_path");
    device_ = yaml_read<std::string>(yaml, "device");
    int x = 0, y = 0, width = 0, height = 0;
    x = yaml["roi"]["x"].as<int>();
    y = yaml["roi"]["y"].as<int>();
    width = yaml["roi"]["width"].as<int>();
    height = yaml["roi"]["height"].as<int>();
    use_roi_ = yaml_read<bool>(yaml, "use_roi");
    roi_ = cv::Rect(x, y, width, height);
    offset_ = cv::Point2f(x, y);

    auto model = core_.read_model(model_path_);
    ov::preprocess::PrePostProcessor ppp(model);
    auto & input = ppp.input();

    input.tensor()
        .set_element_type(ov::element::u8)
        .set_shape({1, 416, 416, 3})
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);

    input.model().set_layout("NCHW");

    input.preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale(255.0);

    // TODO: ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
    model = ppp.build();
    compiled_model_ = core_.compile_model(
        model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

    // 复用 InferRequest 与 letterbox 输入缓冲，避免逐帧创建推理请求和分配内存
    infer_request_ = compiled_model_.create_infer_request();
    input_ = cv::Mat(416, 416, CV_8UC3, cv::Scalar(0, 0, 0));
}

std::vector<Armor> YOLOV8::detect(const cv::Mat & raw_img, int frame_count)
{
    if (raw_img.empty()) {
        logger()->warn("Empty img!, camera drop!");
        return std::vector<Armor>();
    }

    cv::Mat bgr_img;
    if (use_roi_) {
        if (roi_.width == -1) {  // -1 表示该维度不裁切
            roi_.width = raw_img.cols;
        }
        if (roi_.height == -1) {  // -1 表示该维度不裁切
            roi_.height = raw_img.rows;
        }
        bgr_img = raw_img(roi_);
    } else {
        bgr_img = raw_img;
    }

    // preproces（letterbox 见 image 模块）
    double scale;
    letterbox(bgr_img, input_, scale);
    ov::Tensor input_tensor(ov::element::u8, {1, 416, 416, 3}, input_.data);

    /// infer
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();

    // postprocess
    auto output_tensor = infer_request_.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

    return parse(scale, output, raw_img, frame_count);
}

std::vector<Armor> YOLOV8::parse(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
    // 输出为 feature-major 布局 [1][features][anchors]：
    // 直接按步长遍历，省去原先 cv::transpose 的整矩阵拷贝
    const float * data = output.ptr<float>(0);
    const int anchor_num = output.cols;

    std::vector<int> ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> armors_key_points;
    ids.reserve(64);
    confidences.reserve(64);
    boxes.reserve(64);
    armors_key_points.reserve(64);

    for (int r = 0; r < anchor_num; r++) {
        const float * xywh = data + r;
        const float * scores = data + 4 * anchor_num + r;
        const float * key_points = data + (4 + class_num_) * anchor_num + r;

        // 原实现为 cv::minMaxLoc(scores)：取最大分数及其下标
        // 注意 feature-major 布局：类别 c 的分数位于 scores[c * anchor_num]
        float score = scores[0];
        int max_point = 0;
        for (int c = 1; c < class_num_; c++) {
            if (scores[c * anchor_num] > score) {
                score = scores[c * anchor_num];
                max_point = c;
            }
        }

        if (score < score_threshold_) continue;

        auto x = xywh[0 * anchor_num];
        auto y = xywh[1 * anchor_num];
        auto w = xywh[2 * anchor_num];
        auto h = xywh[3 * anchor_num];
        auto left = static_cast<int>((x - 0.5 * w) / scale);
        auto top = static_cast<int>((y - 0.5 * h) / scale);
        auto width = static_cast<int>(w / scale);
        auto height = static_cast<int>(h / scale);

        std::vector<cv::Point2f> armor_key_points;
        armor_key_points.reserve(4);
        for (int i = 0; i < 4; i++) {
            float x = key_points[(i * 2 + 0) * anchor_num] / scale;
            float y = key_points[(i * 2 + 1) * anchor_num] / scale;
            armor_key_points.emplace_back(x, y);
        }
        ids.emplace_back(max_point);
        confidences.emplace_back(score);
        boxes.emplace_back(left, top, width, height);
        armors_key_points.emplace_back(std::move(armor_key_points));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

    std::vector<Armor> armors;
    for (const auto & i : indices) {
        sort_keypoints(armors_key_points[i]);
        if (use_roi_) {
            armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
        } else {
            armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
        }
    }

    for (std::size_t i = 0; i < armors.size();) {
        auto & a = armors[i];
        a.pattern = Detector::get_pattern_neural(bgr_img, a);
        detector_.classify(a);

        if (!detector_.check_name(a, false)) {
            armors.erase(armors.begin() + i);
            continue;
        }

        a.type = Detector::get_type_neural(a);
        if (!Detector::check_type_neural(a)) {
            armors.erase(armors.begin() + i);
            continue;
        }

        a.center_norm = detector_.get_center_norm(bgr_img, a.center);
        ++i;
    }

    if (debug_) draw_detections(bgr_img, armors, frame_count);

    return armors;
}

void YOLOV8::draw_detections(
    const cv::Mat & img, const std::vector<Armor> & armors, int frame_count) const
{
    auto detection = img.clone();
    draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
    for (const auto & armor : armors) {
        auto info = fmt::format(
            "{:.2f} {} {}", armor.confidence, ARMOR_NAMES[armor.name], ARMOR_TYPES[armor.type]);
        draw_points(detection, armor.points, {0, 255, 0});
        draw_text(detection, info, armor.center, {0, 255, 0});
    }

    if (use_roi_) {
        cv::Scalar green(0, 255, 0);
        cv::rectangle(detection, roi_, green, 2);
    }
    cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("detection", detection);
}

void YOLOV8::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
    if (keypoints.size() != 4) {
        std::cout << "beyond 4!!" << std::endl;
        return;
    }

    std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
        return a.y < b.y;
    });

    std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
    std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

    std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
        return a.x < b.x;
    });

    std::sort(
        bottom_points.begin(), bottom_points.end(),
        [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

    keypoints[0] = top_points[0];     // top-left
    keypoints[1] = top_points[1];     // top-right
    keypoints[2] = bottom_points[1];  // bottom-right
    keypoints[3] = bottom_points[0];  // bottom-left
}

std::vector<Armor> YOLOV8::postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
    return parse(scale, output, bgr_img, frame_count);
}

