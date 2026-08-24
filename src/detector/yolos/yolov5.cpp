#include "detector/yolos/yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>

#include "image/image.hpp"
#include "tools/yaml.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"


YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
    auto yaml = yaml_load(config_path);

    model_path_ = yaml_read<std::string>(yaml, "yolov5_model_path");
    device_ = yaml_read<std::string>(yaml, "device");
    use_tensorrt_ = yaml["use_tensorrt"] ? yaml["use_tensorrt"].as<bool>() : false;
    int x = 0, y = 0, width = 0, height = 0;
    x = yaml["roi"]["x"].as<int>();
    y = yaml["roi"]["y"].as<int>();
    width = yaml["roi"]["width"].as<int>();
    height = yaml["roi"]["height"].as<int>();
    use_roi_ = yaml_read<bool>(yaml, "use_roi");
    use_traditional_ = yaml_read<bool>(yaml, "use_traditional");
    roi_ = cv::Rect(x, y, width, height);
    offset_ = cv::Point2f(x, y);

    // 预分配 letterbox 输入缓冲（两后端共用）
    input_ = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    if (use_tensorrt_) {
        // ---- TensorRT 后端 ----
        std::string engine_path = yaml_read<std::string>(yaml, "yolov5_engine_path");
        trt_ = std::make_unique<TensorRTInfer>(engine_path);
        if (!trt_->is_ok()) {
            logger()->error("[YOLOV5] TensorRT 引擎加载失败: {}", engine_path);
            exit(1);
        }
        int64_t vol = 1;
        for (auto d : trt_->input_info().dims) vol *= d;
        blob_.assign(vol, 0.0f);
        logger()->info("[YOLOV5] 使用 TensorRT 后端: {}", engine_path);
        return;
    }

    // ---- OpenVINO 后端 ----
    auto model = core_.read_model(model_path_);
    ov::preprocess::PrePostProcessor ppp(model);
    auto & input = ppp.input();

    input.tensor()
        .set_element_type(ov::element::u8)
        .set_shape({1, 640, 640, 3})
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

    // 复用 InferRequest，避免逐帧创建推理请求
    infer_request_ = compiled_model_.create_infer_request();
}

std::vector<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
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

    if (use_tensorrt_) {
        // TensorRT 预处理：u8 BGR letterbox -> f32 NCHW RGB /255
        // （与 OpenVINO PrePostProcessor 的 convert_color(RGB) + scale(255) 完全一致）
        const int plane = 640 * 640;
        const uchar * src = input_.data;
        float * dst = blob_.data();
        for (int c = 0; c < 3; ++c) {
            const uchar * sc = src + (2 - c);  // BGR -> RGB 通道交换
            float * dc = dst + c * plane;
            for (int i = 0; i < plane; ++i) dc[i] = sc[i * 3] * (1.0f / 255.0f);
        }

        auto & outputs = trt_->infer(blob_.data());
        auto & o = outputs[0];
        int rows = 0, cols = 0;
        if (o.dims.size() == 3) {  // [1, anchors, features]
            rows = static_cast<int>(o.dims[1]);
            cols = static_cast<int>(o.dims[2]);
        } else if (o.dims.size() == 2) {  // [anchors, features]
            rows = static_cast<int>(o.dims[0]);
            cols = static_cast<int>(o.dims[1]);
        } else {
            logger()->error("[YOLOV5] 非法输出维度: {}", o.dims.size());
            return {};
        }
        cv::Mat output(rows, cols, CV_32F, o.data.data());
        return parse(scale, output, raw_img, frame_count);
    }

    // OpenVINO 后端
    ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input_.data);

    // infer
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();

    // postprocess
    auto output_tensor = infer_request_.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

    return parse(scale, output, raw_img, frame_count);
}

std::vector<Armor> YOLOV5::parse(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
    // 输出为 anchor-major 布局 [1][anchors][22]，每行 22 个特征连续存放：
    // 直接按行遍历，省去逐行 Mat 视图与 minMaxLoc 的开销
    const float * data = output.ptr<float>(0);
    const int feature_num = output.cols;  // 22
    const int anchor_num = output.rows;

    std::vector<int> color_ids, num_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> armors_key_points;
    color_ids.reserve(64);
    num_ids.reserve(64);
    confidences.reserve(64);
    boxes.reserve(64);
    armors_key_points.reserve(64);

    for (int r = 0; r < anchor_num; r++) {
        const float * row = data + r * feature_num;

        double score = row[8];
        score = sigmoid(score);

        if (score < score_threshold_) continue;

        std::vector<cv::Point2f> armor_key_points;
        armor_key_points.reserve(4);

        // 颜色(9..12)与类别(13..21)独热向量取 argmax（下标为局部下标，与原实现一致）
        double score_num = row[13], score_color = row[9];
        int _class_id = 0, _color_id = 0;
        for (int c = 1; c < 9; c++) {
            if (row[13 + c] > score_num) {
                score_num = row[13 + c];
                _class_id = c;
            }
        }
        for (int c = 1; c < 4; c++) {
            if (row[9 + c] > score_color) {
                score_color = row[9 + c];
                _color_id = c;
            }
        }

        armor_key_points.emplace_back(row[0] / scale, row[1] / scale);
        armor_key_points.emplace_back(row[6] / scale, row[7] / scale);
        armor_key_points.emplace_back(row[4] / scale, row[5] / scale);
        armor_key_points.emplace_back(row[2] / scale, row[3] / scale);

        float min_x = armor_key_points[0].x;
        float max_x = armor_key_points[0].x;
        float min_y = armor_key_points[0].y;
        float max_y = armor_key_points[0].y;

        for (int i = 1; i < 4; i++) {
            if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
            if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
            if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
            if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
        }

        cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

        color_ids.emplace_back(_color_id);
        num_ids.emplace_back(_class_id);
        boxes.emplace_back(rect);
        confidences.emplace_back(static_cast<float>(score));
        armors_key_points.emplace_back(std::move(armor_key_points));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

    std::vector<Armor> armors;
    for (const auto & i : indices) {
        if (use_roi_) {
            armors.emplace_back(
                color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
        } else {
            armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
        }
    }

    for (std::size_t i = 0; i < armors.size();) {
        auto & a = armors[i];
        if (!detector_.check_name(a, false)) {
            armors.erase(armors.begin() + i);
            continue;
        }

        if (!Detector::check_type_neural(a)) {
            armors.erase(armors.begin() + i);
            continue;
        }
        // 使用传统方法二次矫正角点
        if (use_traditional_) detector_.detect(a, bgr_img);

        a.center_norm = detector_.get_center_norm(bgr_img, a.center);
        ++i;
    }

    if (debug_) draw_detections(bgr_img, armors, frame_count);

    return armors;
}

void YOLOV5::draw_detections(
    const cv::Mat & img, const std::vector<Armor> & armors, int frame_count) const
{
    auto detection = img.clone();
    draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
    for (const auto & armor : armors) {
        auto info = fmt::format(
            "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
            ARMOR_TYPES[armor.type]);
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

double YOLOV5::sigmoid(double x)
{
    if (x > 0)
        return 1.0 / (1.0 + exp(-x));
    else
        return exp(x) / (1.0 + exp(x));
}

std::vector<Armor> YOLOV5::postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
    return parse(scale, output, bgr_img, frame_count);
}

